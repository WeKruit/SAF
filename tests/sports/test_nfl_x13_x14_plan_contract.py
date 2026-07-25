from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import yaml

from prediction_market.experiments import load_experiment_registry
from prediction_market.sports.nfl_x13_association import (
    X13_REGISTERED_ANALYSIS_LOCK_V1,
)
from prediction_market.sports.nfl_x13_spec import (
    X13_GAME_IDS,
    X13_HORIZONS_SECONDS,
)


ROOT = Path(__file__).resolve().parents[2]
X13_CACHE = (
    ROOT
    / "artifacts/market-observation/nfl/x13/correlation-cache"
    / "sha256-48d1f35e0e7fc63dc1a75611da6e81b2834c69d2d7fe6be4af4d9e3074e6d62e"
)
X13_SIGNAL_ROOT = ROOT / "artifacts/market-observation/nfl/x13/signal-exploration"


def _yaml_card(experiment_id: str) -> dict[str, object]:
    path = ROOT / f"registries/experiments/{experiment_id}.yaml"
    assert path.is_file(), f"{experiment_id} must be preregistered"
    value = yaml.safe_load(path.read_text("utf-8"))
    assert isinstance(value, dict)
    card = value.get("experiment_card")
    assert isinstance(card, dict)
    return card


def _signal_bundles() -> list[tuple[Path, dict[str, object]]]:
    bundles: list[tuple[Path, dict[str, object]]] = []
    for manifest_path in sorted(X13_SIGNAL_ROOT.glob("sha256-*/*.manifest.json")):
        value = json.loads(manifest_path.read_bytes())
        if (
            value.get("artifact_type")
            == "nfl_x13_signal_exploration_bundle"
            and value.get("result_label") == "PRELIMINARY_SOURCE_TIME_ONLY"
        ):
            bundles.append((manifest_path.parent, value))
    assert bundles, "X-13 must publish a verified signal bundle"
    return bundles


def _logical_file(
    root: Path,
    manifest: dict[str, object],
    logical_name: str,
) -> Path | None:
    files = manifest.get("files")
    assert isinstance(files, list)
    matches = [
        item
        for item in files
        if isinstance(item, dict) and item.get("logical_name") == logical_name
    ]
    if len(matches) != 1:
        return None
    relative = matches[0].get("path")
    assert isinstance(relative, str)
    return root / relative


def test_x13_exact_games_horizons_and_observed_trade_no_ffill_contract() -> None:
    assert len(X13_GAME_IDS) == 20
    assert X13_HORIZONS_SECONDS == (1, 2, 5, 10, 30, 60)
    observed = X13_CACHE / "x13_reaction_observed_cache_v1.parquet"
    connection = duckdb.connect()
    try:
        game_ids, horizons, invalid_rows = connection.execute(
            """
            SELECT
              list(DISTINCT game_id ORDER BY game_id),
              list(DISTINCT horizon_seconds ORDER BY horizon_seconds),
              count(*) FILTER (
                WHERE validity_status != 'OBSERVED'
                   OR trade_count <= 0
                   OR pre_event_actual_trade IS NULL
                   OR first_post_event_trade IS NULL
              )
            FROM read_parquet(?)
            """,
            [str(observed)],
        ).fetchone()
    finally:
        connection.close()
    assert tuple(game_ids) == X13_GAME_IDS
    assert tuple(horizons) == X13_HORIZONS_SECONDS
    assert invalid_rows == 0, "missing horizons must not be forward-filled"


def test_signal_bundle_publishes_per_game_market_coverage() -> None:
    required = {
        "game_id",
        "venue",
        "market_kind",
        "observation_kind",
        "provenance_status",
        "episode_count",
        "contract_count",
        "actual_observation_count",
        "observed_horizon_count",
    }
    for root, manifest in _signal_bundles():
        path = _logical_file(root, manifest, "per_game_market_coverage")
        if path is None or not required <= set(pq.read_schema(path).names):
            continue
        connection = duckdb.connect()
        try:
            games, actual_games, candle_games, invalid = connection.execute(
                """
                SELECT
                  list(DISTINCT game_id ORDER BY game_id),
                  count(DISTINCT game_id) FILTER (
                    WHERE observation_kind = 'actual_trade'
                  ),
                  count(DISTINCT game_id) FILTER (
                    WHERE venue = 'kalshi'
                      AND observation_kind = 'kalshi_1m_candle_bbo'
                  ),
                  count(*) FILTER (
                    WHERE (
                      observation_kind = 'actual_trade'
                      AND (
                        provenance_status != 'BOUND_OBSERVED_CACHE'
                        OR actual_observation_count <= 0
                      )
                    )
                    OR (
                      observation_kind = 'kalshi_1m_candle_bbo'
                      AND (
                        venue != 'kalshi'
                        OR provenance_status != 'NOT_CARRIED_TO_SIGNAL_CACHE'
                        OR actual_observation_count != 0
                      )
                    )
                  )
                FROM read_parquet(?)
                """,
                [str(path)],
            ).fetchone()
        finally:
            connection.close()
        if (
            tuple(games) == X13_GAME_IDS
            and actual_games == 20
            and candle_games == 20
            and invalid == 0
        ):
            return
    raise AssertionError(
        "signal bundle needs immutable per-game observed-market coverage"
    )


def test_signal_bundle_publishes_twenty_game_actual_paths_not_only_pooled() -> None:
    for root, manifest in _signal_bundles():
        path = _logical_file(root, manifest, "actual_observation_paths")
        if path is None:
            continue
        required = {
            "game_id",
            "episode_id",
            "contract_id",
            "outcome",
            "venue",
            "family",
            "change_1s",
            "change_2s",
            "change_5s",
            "change_10s",
            "change_30s",
            "change_60s",
            "observed_1s",
            "observed_2s",
            "observed_5s",
            "observed_10s",
            "observed_30s",
            "observed_60s",
            "path_complete",
            "path_factor_value",
        }
        if not required <= set(pq.read_schema(path).names):
            continue
        connection = duckdb.connect()
        try:
            games, forward_filled = connection.execute(
                """
                SELECT
                  list(DISTINCT game_id ORDER BY game_id),
                  count(*) FILTER (
                    WHERE (NOT observed_1s AND change_1s IS NOT NULL)
                       OR (NOT observed_2s AND change_2s IS NOT NULL)
                       OR (NOT observed_5s AND change_5s IS NOT NULL)
                       OR (NOT observed_10s AND change_10s IS NOT NULL)
                       OR (NOT observed_30s AND change_30s IS NOT NULL)
                       OR (NOT observed_60s AND change_60s IS NOT NULL)
                  )
                FROM read_parquet(?)
                """,
                [str(path)],
            ).fetchone()
        finally:
            connection.close()
        if tuple(games) == X13_GAME_IDS and forward_filled == 0:
            return
    raise AssertionError(
        "signal publication needs one non-pooled actual_observation_paths artifact"
    )


def test_every_frozen_hypothesis_has_an_explicit_result_or_not_run_row() -> None:
    expected = {
        hypothesis.hypothesis_id
        for hypothesis in X13_REGISTERED_ANALYSIS_LOCK_V1.hypotheses
    }
    assert expected == {
        "event_superclass",
        "fair_delta_by_both_venues_active",
        "fair_delta_by_late_close_state",
        "fair_delta_by_pre_event_liquidity",
        "fair_delta_by_semantic_salience",
        "sequence",
        "signed_diagnostic_fair_delta_by_venue",
        "signed_diagnostic_fair_delta_pooled",
    }
    for root, manifest in _signal_bundles():
        path = _logical_file(root, manifest, "registered_hypothesis_results")
        if path is None or "hypothesis_id" not in pq.read_schema(path).names:
            continue
        connection = duckdb.connect()
        try:
            observed = set(
                connection.execute(
                    "SELECT DISTINCT hypothesis_id FROM read_parquet(?)",
                    [str(path)],
                )
                .fetchnumpy()["hypothesis_id"]
                .tolist()
            )
        finally:
            connection.close()
        if observed == expected:
            return
    raise AssertionError(
        "all frozen hypothesis families need explicit result or NOT_RUN rows"
    )


def test_signal_bundle_is_registered_before_any_candidate_interpretation() -> None:
    x13 = _yaml_card("X-13")
    registered_results: set[str] = set()
    for item in x13.get("results_ref", []):
        if isinstance(item, dict) and isinstance(item.get("result_sha256"), str):
            registered_results.add(item["result_sha256"])
    for amendment in x13.get("amendments", []):
        if not isinstance(amendment, dict):
            continue
        changes = amendment.get("changes")
        if not isinstance(changes, dict):
            continue
        result = changes.get("results_ref")
        if isinstance(result, dict) and isinstance(
            result.get("result_sha256"), str
        ):
            registered_results.add(result["result_sha256"])
    published = {
        manifest["bundle_sha256"] for _, manifest in _signal_bundles()
    }
    assert published <= registered_results, (
        "post-hoc signal bundles must be registry-bound before interpretation"
    )


def test_pooled_inference_keeps_cluster_loo_concentration_bh_and_cross_venue() -> None:
    root, manifest = _signal_bundles()[-1]
    primary = _logical_file(root, manifest, "primary_contrasts")
    venue = _logical_file(root, manifest, "cross_venue_symmetry")
    assert primary is not None and venue is not None
    assert {
        "ci_low",
        "ci_high",
        "logo_sign_stability_rate",
        "maximum_single_game_contribution",
        "bootstrap_samples_requested",
        "bootstrap_samples_valid",
        "bh_q_value",
    } <= set(pq.read_schema(primary).names)
    assert {
        "shared_game_count",
        "mean_signed_price_change_a",
        "mean_signed_price_change_b",
        "sign_agreement_rate",
    } <= set(pq.read_schema(venue).names)


def test_x13_selector_and_x14_prospective_confirmatory_roles_are_separate() -> None:
    x13 = load_experiment_registry(ROOT)["X-13"]
    selector = x13.get("historical_holdout_selector")
    assert isinstance(selector, dict)
    assert selector.get("owner_experiment_id") == "X-13"
    assert selector.get("selection_inputs_exclude_market_reaction") is True
    assert selector.get("frozen_before_reaction_access") is True

    card = _yaml_card("X-14")
    assert card.get("id") == "X-14"
    protocol = card.get("prospective_confirmatory_protocol")
    assert isinstance(protocol, dict)
    assert protocol.get("role") == "PROSPECTIVE_CONFIRMATORY"
    assert protocol.get("discovery_experiment_id") == "X-13"
    assert protocol.get("recorder_continuity_experiment_id") == "X-08"
    assert protocol.get("historical_holdout_selector_owner") == "X-13"
    assert protocol.get("prediction_commit_precedes_reaction_read") is True
    assert protocol.get("reaction_data_access_before_prediction_commit") is False


def test_x13_historical_and_x08_prospective_scopes_remain_separate() -> None:
    x13 = _yaml_card("X-13")
    x08 = _yaml_card("X-08")
    assert x13.get("source_time_only") is True
    assert x13.get("causal_or_execution_claims_authorized") is False
    prospective = x08.get("prospective_observation")
    assert isinstance(prospective, dict)
    assert prospective.get("required_elapsed_days") == 7
    assert prospective.get("fixtures_can_satisfy_elapsed_time") is False
    assert x13.get("id") != x08.get("id")


def test_x13_signal_claim_boundary_stays_preliminary() -> None:
    root, manifest = _signal_bundles()[-1]
    report_path = _logical_file(root, manifest, "report")
    assert report_path is not None
    report = json.loads(report_path.read_bytes())
    assert report["result_label"] == "PRELIMINARY_SOURCE_TIME_ONLY"
    assert report["analysis_status"] == "POST_HOC_DISCOVERY_HOLDOUT_REQUIRED"
    expected_boundary = {
        "alpha": False,
        "causal": False,
        "execution": False,
        "latency": False,
        "tradeability": False,
    }
    assert {
        key: report["claim_boundary"][key]
        for key in expected_boundary
    } == expected_boundary
    assert report["claim_boundary"]["depth"] is False
    assert report["claim_boundary"]["order_flow_imbalance"] is False
