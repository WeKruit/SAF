from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from prediction_market.sports.nfl_x13_capture import (
    GIB,
    CaptureHttpResponse,
    CaptureTarget,
    capture_x13_batch,
    load_x13_capture,
)
from prediction_market.sports.nfl_x13_dataset import X13SourceBundle
from prediction_market.sports.nfl_x13_game_build import (
    X13GameStateArtifact,
    X13GameStateBuild,
    build_x13_game_state_manifest,
    publish_x13_game_state,
)
from prediction_market.sports.nfl_x13_replay import build_x13_game_ledger
from prediction_market.sports.nfl_x13_spec import (
    X13_BATCH_SPEC,
    X13_GAME_BINDINGS,
    X13_GAME_IDS,
    canonical_sha256,
)
from prediction_market.sports.nfl_x13_universe import (
    KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1,
    KalshiSeriesRegistryProofV1,
    PolymarketTokenBindingV1,
    SourceTimeWindowV1,
    UniverseContractV1,
    UniverseGameV1,
    X13UniverseBatchV1,
    publish_x13_universe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc
NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
SHA_A = f"sha256:{'a' * 64}"
SHA_B = f"sha256:{'b' * 64}"
SHA_C = f"sha256:{'c' * 64}"
EXPECTED_X13_EVALUATION_CODE_PATHS = (
    "src/prediction_market/experiments.py",
    "src/prediction_market/sports/nfl_x13_association.py",
    "src/prediction_market/sports/nfl_x13_batch.py",
    "src/prediction_market/sports/nfl_x13_capture.py",
    "src/prediction_market/sports/nfl_x13_dataset.py",
    "src/prediction_market/sports/nfl_x13_game_build.py",
    "src/prediction_market/sports/nfl_x13_market.py",
    "src/prediction_market/sports/nfl_x13_pipeline.py",
    "src/prediction_market/sports/nfl_x13_replay.py",
    "src/prediction_market/sports/nfl_x13_spec.py",
    "src/prediction_market/sports/nfl_x13_universe.py",
)


def _source_second(index: int) -> datetime:
    binding = X13_GAME_BINDINGS[index]
    return datetime.combine(binding.game_date, time(17, 0), tzinfo=UTC)


def _fixture_game_build() -> X13GameStateBuild:
    games: dict[str, X13GameStateArtifact] = {}
    pbp: dict[str, tuple[dict[str, object], ...]] = {}
    participation: dict[str, tuple[dict[str, object], ...]] = {}
    for index, binding in enumerate(X13_GAME_BINDINGS):
        source_at = _source_second(index)
        live = {
            "game_id": binding.native_game_id,
            "play_id": "1",
            "order_sequence": 1,
            "raw_ordinal": 0,
            "time_of_day": source_at.isoformat(),
            "source_time_end_utc": (source_at + timedelta(seconds=1)).isoformat(),
            "qtr": 1,
            "quarter_seconds_remaining": 900,
            "game_seconds_remaining": 3600,
            "home_team": binding.home_team,
            "away_team": binding.away_team,
            "home_score": 0,
            "away_score": 0,
            "posteam": binding.away_team,
            "defteam": binding.home_team,
            "fixed_drive": 1,
            "down": 1,
            "ydstogo": 10,
            "yardline_100": 75,
            "direction": "right",
            "field_coordinate_0_100": 25,
            "field_orientation_semantics": (
                "canonical_team_end_orientation_not_tracking_coordinate"
            ),
            "play_type": "run",
            "rush_attempt": 1,
            "rusher_player_name": f"{binding.away_team} Runner",
            "rushing_yards": 5,
            "desc": "Run for five yards.",
            "total_home_score_post": 0,
            "total_away_score_post": 0,
            "game_end": 0,
        }
        terminal_at = source_at + timedelta(seconds=100)
        terminal = {
            "game_id": binding.native_game_id,
            "play_id": "2",
            "order_sequence": 2,
            "raw_ordinal": 1,
            "time_of_day": terminal_at.isoformat(),
            "source_time_end_utc": (
                terminal_at + timedelta(seconds=1)
            ).isoformat(),
            "qtr": 4,
            "quarter_seconds_remaining": 0,
            "game_seconds_remaining": 0,
            "home_team": binding.home_team,
            "away_team": binding.away_team,
            "home_score": binding.home_score,
            "away_score": binding.away_score,
            "total_home_score_post": binding.home_score,
            "total_away_score_post": binding.away_score,
            "desc": "END GAME",
            "game_end": 1,
        }
        personnel_rows = (
            {
                "nflverse_game_id": binding.native_game_id,
                "play_id": "1",
                "offense_formation": "SHOTGUN",
                "offense_personnel": "11",
                "defense_personnel": "nickel",
                "offense_names": ";".join(
                    f"{binding.away_team} O{number}" for number in range(11)
                ),
                "defense_names": ";".join(
                    f"{binding.home_team} D{number}" for number in range(11)
                ),
            },
        )
        source_rows = (live, terminal)
        ledger = build_x13_game_ledger(source_rows, personnel_rows)
        games[binding.native_game_id] = X13GameStateArtifact(
            game_id=binding.native_game_id,
            events=ledger.events,
            episodes=ledger.episodes,
            stat_ledger=ledger.stat_ledger,
            events_sha256=ledger.events_sha256,
            ledger_sha256=ledger.artifact_sha256,
            source_time_present_rows=2,
            participation_status_counts=MappingProxyType(
                {"missing": 1, "present_complete": 1}
            ),
            final_score=MappingProxyType(
                {
                    binding.away_team: binding.away_score,
                    binding.home_team: binding.home_score,
                }
            ),
            final_score_verified=True,
            personnel_summary=MappingProxyType(
                {
                    "license_status": "research_only",
                    "semantics": (
                        "observed on-field roster per play; adjacent set differences "
                        "are between_play_roster_change, not exact substitution time"
                    ),
                    "source_participation_rows": 1,
                    "present_complete_11v11_rows": 1,
                    "present_partial_rows": 0,
                    "missing_rows": 1,
                }
            ),
            runtime_ns=1,
        )
        pbp[binding.native_game_id] = source_rows
        participation[binding.native_game_id] = personnel_rows
    source = X13SourceBundle(
        pbp_rows_by_game=MappingProxyType(pbp),
        participation_rows_by_game=MappingProxyType(participation),
        pbp_object_sha256=SHA_A,
        pbp_manifest_sha256=SHA_B,
        participation_object_sha256=SHA_C,
        participation_manifest_sha256=f"sha256:{'d' * 64}",
        selected_rows_sha256=f"sha256:{'e' * 64}",
        sample_audit=MappingProxyType({"game_count": 20, "row_count": 40}),
        full_11v11_participation_rows=20,
    )
    census = {"fixture": True, "transitions": 1}
    manifest = build_x13_game_state_manifest(
        ledger_hashes={
            game_id: games[game_id].ledger_sha256 for game_id in X13_GAME_IDS
        },
        source_manifest_hashes=(
            source.pbp_manifest_sha256,
            source.participation_manifest_sha256,
        ),
        census_sha256=canonical_sha256(census),
        builder_version="nfl-x13-game-state-v1",
        runtime={
            "elapsed_ns": 1,
            "peak_rss_bytes": 1,
            "compute_workers": 2,
            "census_transition_count": 1,
            "census_latency_p50_ns": 1,
            "census_latency_p95_ns": 1,
            "census_latency_p99_ns": 1,
            "ledger_runtime_total_ns": 1,
        },
    )
    return X13GameStateBuild(
        games=MappingProxyType(games),
        census=MappingProxyType(census),
        source=source,
        manifest=MappingProxyType(manifest),
    )


def _initial_universe() -> X13UniverseBatchV1:
    all_series = tuple(
        item.series_ticker
        for item in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
    )
    games: list[UniverseGameV1] = []
    for index, binding in enumerate(X13_GAME_BINDINGS):
        source_at = _source_second(index)
        start_ts = int((source_at - timedelta(seconds=60)).timestamp())
        end_ts = int((source_at + timedelta(seconds=120)).timestamp())
        poly_target = CaptureTarget(
            contract_id=f"polymarket:{binding.polymarket_condition_id}",
            venue="polymarket",
            venue_market_id=binding.polymarket_market_id,
            condition_id=binding.polymarket_condition_id,
            venue_title=f"{binding.away_team} vs. {binding.home_team}",
            kalshi_series_ticker=None,
            kalshi_event_ticker=None,
            family="moneyline",
            dependency_game_ids=(binding.native_game_id,),
            kind="primitive",
            analysis_eligible=True,
        )
        kalshi_targets = tuple(
            CaptureTarget(
                contract_id=f"kalshi:{ticker}",
                venue="kalshi",
                venue_market_id=ticker,
                condition_id=None,
                venue_title=f"{binding.away_team} at {binding.home_team}",
                kalshi_series_ticker="KXNFLGAME",
                kalshi_event_ticker=binding.kalshi_event_ticker,
                family="moneyline",
                dependency_game_ids=(binding.native_game_id,),
                kind="primitive",
                analysis_eligible=True,
            )
            for ticker in (
                binding.kalshi_away_ticker,
                binding.kalshi_home_ticker,
            )
        )
        poly_contract = UniverseContractV1(
            contract_id=poly_target.contract_id,
            venue_market_id=binding.polymarket_market_id,
            raw_contract_ids=(poly_target.contract_id,),
            condition_id=binding.polymarket_condition_id,
            logical_market_id=f"polymarket:{binding.native_game_id}:winner",
            venue="polymarket",
            family="moneyline",
            period="game",
            subject=f"{binding.away_team} at {binding.home_team}",
            measure="winner",
            comparator="eq",
            line=None,
            outcomes=(binding.away_team, binding.home_team),
            raw_outcome_labels=(),
            rule_sha256=SHA_A,
            dependency_game_ids=(binding.native_game_id,),
            kind="primitive",
            analysis_eligible=True,
            outcome_coverage=(
                (binding.away_team, "unavailable_no_trades"),
                (binding.home_team, "unavailable_no_trades"),
            ),
        )
        kalshi_contract = UniverseContractV1(
            contract_id=f"kalshi:{binding.kalshi_event_ticker}",
            venue_market_id=binding.kalshi_event_ticker,
            raw_contract_ids=(
                binding.kalshi_away_ticker,
                binding.kalshi_home_ticker,
            ),
            condition_id=None,
            logical_market_id=f"kalshi:{binding.native_game_id}:winner",
            venue="kalshi",
            family="moneyline",
            period="game",
            subject=f"{binding.away_team} at {binding.home_team}",
            measure="winner",
            comparator="eq",
            line=None,
            outcomes=(binding.away_team, binding.home_team),
            raw_outcome_labels=(
                (binding.kalshi_away_ticker, binding.away_team, binding.home_team),
                (binding.kalshi_home_ticker, binding.home_team, binding.away_team),
            ),
            rule_sha256=SHA_B,
            dependency_game_ids=(binding.native_game_id,),
            kind="primitive",
            analysis_eligible=True,
            outcome_coverage=(
                (binding.away_team, "unavailable_no_trades"),
                (binding.home_team, "unavailable_no_trades"),
            ),
        )
        games.append(
            UniverseGameV1(
                game_id=binding.native_game_id,
                source_window=SourceTimeWindowV1(start_ts, end_ts),
                polymarket_event_id=binding.polymarket_event_id,
                polymarket_event_slug=(
                    f"nfl-{binding.away_team.lower()}-"
                    f"{binding.home_team.lower()}-{binding.game_date.isoformat()}"
                ),
                polymarket_event_title=(
                    f"{binding.away_team} vs. {binding.home_team}"
                ),
                primitive_targets=(poly_target, *kalshi_targets),
                same_game_composite_targets=(),
                cross_game_inventory=(),
                excluded_contracts=(),
                unknown_contract_ids=(),
                contracts=(kalshi_contract, poly_contract),
                polymarket_token_bindings=(
                    PolymarketTokenBindingV1(
                        condition_id=binding.polymarket_condition_id,
                        outcomes=(binding.away_team, binding.home_team),
                        token_ids=(
                            f"token-{index}-away",
                            f"token-{index}-home",
                        ),
                    ),
                ),
                kalshi_series_tickers=all_series,
                metadata_manifest_sha256s=(SHA_A,),
            )
        )
    games_tuple = tuple(games)
    return X13UniverseBatchV1(
        games=games_tuple,
        capture_plans=tuple(game.capture_plan for game in games_tuple),
        series_registry_proof=KalshiSeriesRegistryProofV1(
            series_tickers=all_series,
            manifest_sha256=SHA_A,
        ),
    )


def _json_response(value: object) -> CaptureHttpResponse:
    return CaptureHttpResponse(
        body=json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        headers=(("Content-Type", "application/json"),),
    )


def _capture_fixture(
    universe: X13UniverseBatchV1,
    raw_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import prediction_market.sports.nfl_x13_capture as capture_module

    monkeypatch.setattr(
        capture_module,
        "X13_FROZEN_UNIVERSE_ID",
        universe.universe_id,
    )
    monkeypatch.setattr(
        capture_module,
        "X13_FROZEN_CAPTURE_TARGET_COUNT",
        sum(len(plan.targets) for plan in universe.capture_plans),
    )
    game_by_id = {game.game_id: game for game in universe.games}
    index_by_game = {
        binding.native_game_id: index
        for index, binding in enumerate(X13_GAME_BINDINGS)
    }

    def requester(request):
        if request.resource == "kalshi_cutoff":
            return _json_response(
                {
                    "market_positions_last_updated_ts": "2026-05-25T00:00:00Z",
                    "market_settled_ts": "2026-05-25T00:00:00Z",
                    "orders_updated_ts": "2026-05-25T00:00:00Z",
                    "trades_created_ts": "2026-05-25T00:00:00Z",
                }
            )
        game = game_by_id[request.game_id]
        binding = X13_GAME_BINDINGS[index_by_game[request.game_id]]
        source_at = _source_second(index_by_game[request.game_id])
        params = dict(request.params)
        if request.resource == "gamma_event":
            return _json_response(
                [
                    {
                        "id": game.polymarket_event_id,
                        "slug": game.polymarket_event_slug,
                        "title": game.polymarket_event_title,
                        "markets": [
                            {
                                "id": binding.polymarket_market_id,
                                "conditionId": binding.polymarket_condition_id,
                                "question": game.polymarket_event_title,
                            }
                        ],
                    }
                ]
            )
        if request.resource == "kalshi_markets":
            return _json_response(
                {
                    "markets": [
                        {
                            "ticker": target.venue_market_id,
                            "event_ticker": target.kalshi_event_ticker,
                            "series_ticker": "KXNFLGAME",
                            "title": target.venue_title,
                        }
                        for target in game.capture_targets
                        if target.venue == "kalshi"
                    ],
                    "cursor": "",
                }
            )
        if request.resource == "polymarket_trades":
            return _json_response(
                [
                    {
                        "conditionId": binding.polymarket_condition_id,
                        "asset": (
                            f"token-{index_by_game[request.game_id]}-{side}"
                        ),
                        "outcome": outcome,
                        "timestamp": int(source_at.timestamp()) + offset,
                        "transactionHash": (
                            f"0xpoly-{request.game_id}-{side}-{offset}"
                        ),
                        "price": price,
                        "size": 2.5,
                    }
                    for offset, away_price in (
                        (-2, 0.50),
                        (2, 0.60),
                    )
                    for side, outcome, price in (
                        ("away", binding.away_team, away_price),
                        (
                            "home",
                            binding.home_team,
                            float(Decimal(1) - Decimal(str(away_price))),
                        ),
                    )
                    if params["start"]
                    <= int(source_at.timestamp()) + offset
                    <= params["end"]
                ]
            )
        if request.resource == "kalshi_trades":
            ticker = request.scope_id
            outcome = (
                binding.away_team
                if ticker == binding.kalshi_away_ticker
                else binding.home_team
            )
            return _json_response(
                {
                    "trades": [
                        {
                            "trade_id": f"{ticker}-trade",
                            "ticker": ticker,
                            "created_time": (
                                source_at + timedelta(seconds=2)
                            ).isoformat().replace("+00:00", "Z"),
                        "yes_price_dollars": (
                            "0.58" if outcome == binding.away_team else "0.42"
                        ),
                        "no_price_dollars": (
                            "0.42" if outcome == binding.away_team else "0.58"
                        ),
                        "count_fp": "3",
                        "is_block_trade": outcome == binding.home_team,
                        "taker_side": "yes",
                        "taker_outcome_side": "yes",
                        }
                    ],
                    "cursor": "",
                }
            )
        if request.resource == "kalshi_candlesticks":
            ticker = request.scope_id
            return _json_response(
                {
                    "ticker": ticker,
                    "candlesticks": [
                        {
                            "end_period_ts": int(
                                (source_at + timedelta(seconds=60)).timestamp()
                            ),
                            "yes_bid": {
                                "open": "0.54",
                                "high": "0.56",
                                "low": "0.53",
                                "close": "0.55",
                            },
                            "yes_ask": {
                                "open": "0.56",
                                "high": "0.58",
                                "low": "0.55",
                                "close": "0.57",
                            },
                            "volume": "10",
                            "open_interest": "20",
                        }
                    ],
                }
            )
        raise AssertionError(request.resource)

    return capture_x13_batch(
        universe.capture_plans,
        universe_id=universe.universe_id,
        raw_store_root=raw_root,
        program_root=PROJECT_ROOT,
        planned_raw_p95_bytes=0,
        derived_p95_bytes=0,
        requester=requester,
        response_clock=lambda: NOW,
        host_requests_per_second={
            "gamma-api.polymarket.com": 1_000_000_000,
            "data-api.polymarket.com": 1_000_000_000,
            "external-api.kalshi.com": 1_000_000_000,
        },
        disk_usage=lambda _: SimpleNamespace(free=8 * GIB),
    )

def test_current_registry_blocks_pipeline_before_unresolved_source_bundle() -> None:
    from prediction_market.sports.nfl_x13_pipeline import (
        X13PipelineError,
        load_x13_pipeline_authorization,
    )

    with pytest.raises(X13PipelineError, match="source_manifest_bundle"):
        load_x13_pipeline_authorization(
            PROJECT_ROOT,
            game_state_build_id=SHA_A,
            universe_id=SHA_B,
            capture_sha256=SHA_C,
        )


def test_authorization_lock_binds_effective_amendment_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prediction_market.sports.nfl_x13_pipeline as pipeline_module

    from prediction_market.sports.nfl_x13_pipeline import (
        X13_EVALUATION_CODE_PATHS,
        X13PipelineError,
        expected_analysis_lock_sha256,
        expected_x13_source_manifest_amendment_changes,
        expected_x13_evaluation_code_bundle_sha256,
        expected_source_manifest_bundle_sha256,
        load_x13_pipeline_authorization,
    )

    required_locks = (
        "analysis_whitelist",
        "contract_universe",
        "game_sample_and_bindings",
        "h_split_approval",
        "source_manifest_bundle",
        "source_time_interval_semantics",
    )
    amendment_head = f"sha256:{'d' * 64}"
    game_state_build_id = f"sha256:{'e' * 64}"
    capture_sha256 = f"sha256:{'f' * 64}"
    source_manifest_bundle_sha256 = (
        expected_source_manifest_bundle_sha256(
            plan_id=X13_BATCH_SPEC.plan_id,
            game_state_build_id=game_state_build_id,
            universe_id=SHA_B,
            capture_sha256=capture_sha256,
        )
    )
    expected_changes = expected_x13_source_manifest_amendment_changes(
        program_root=PROJECT_ROOT,
        game_state_build_id=game_state_build_id,
        universe_id=SHA_B,
        capture_sha256=capture_sha256,
    )
    expected_input = expected_changes["preregistered_inputs"][0]
    expected_code_files = [
        {
            "path": relative_path,
            "sha256": "sha256:"
            + hashlib.sha256(
                (PROJECT_ROOT / relative_path).read_bytes()
            ).hexdigest(),
        }
        for relative_path in EXPECTED_X13_EVALUATION_CODE_PATHS
    ]
    assert X13_EVALUATION_CODE_PATHS == (
        EXPECTED_X13_EVALUATION_CODE_PATHS
    )
    assert "plan_id" not in inspect.signature(
        expected_x13_source_manifest_amendment_changes
    ).parameters
    assert expected_x13_evaluation_code_bundle_sha256(
        PROJECT_ROOT
    ) == canonical_sha256(
        {
            "schema": "nfl_x13_evaluation_code_bundle_v1",
            "experiment_id": "X-13",
            "files": expected_code_files,
        }
    )
    assert expected_changes == {
        "resolve_locks": [
            {
                "lock_id": "source_manifest_bundle",
                "evidence_ref": source_manifest_bundle_sha256,
            }
        ],
        "preregistered_inputs": [
            {
                "scope": "preliminary_source_time_only",
                "code_sha256": (
                    expected_x13_evaluation_code_bundle_sha256(PROJECT_ROOT)
                ),
                "data_sha256": source_manifest_bundle_sha256,
                "dataset_ids": [
                    "DS-KALSHI-HISTORICAL",
                    "DS-NFLVERSE",
                    "DS-NFLVERSE-PARTICIPATION",
                    "DS-POLYMARKET-PUBLIC",
                ],
                "model_ids": [],
            }
        ],
    }
    card: dict[str, object] = {
        "status": "registered",
        "execution_authorized": True,
        "source_time_only": True,
        "causal_or_execution_claims_authorized": False,
        "authorization_scopes": {
            "preliminary_source_time_only": {
                "authorized": True,
                "required_result_label": "PRELIMINARY",
                "required_lock_ids": list(required_locks),
            }
        },
        "registration_locks": [
            {
                "id": lock_id,
                "status": "resolved",
                **(
                    {"evidence_ref": source_manifest_bundle_sha256}
                    if lock_id == "source_manifest_bundle"
                    else {}
                ),
            }
            for lock_id in required_locks
        ],
        "registration_record_sha256": SHA_A,
        "registration_head_sha256": amendment_head,
        "preregistered_inputs": {
            "preliminary_source_time_only": {
                **{
                    key: value
                    for key, value in expected_input.items()
                    if key != "scope"
                },
                "registered_at": "2026-07-25T05:00:00Z",
            }
        },
    }
    reviews = [
        SimpleNamespace(
            catalog_item_id=license_ref,
            status="GREEN",
            operational_use=operational_use,
        )
        for license_ref, operational_use in (
            ("I-018", "APPROVED"),
            ("O-001", "RESEARCH_ONLY"),
            ("O-003", "RESEARCH_ONLY"),
            ("O-009", "RESEARCH_ONLY"),
        )
    ]
    monkeypatch.setattr(
        pipeline_module,
        "load_experiment_registry",
        lambda _: {"X-13": card},
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_data_license_register",
        lambda _: reviews,
    )

    authorization = load_x13_pipeline_authorization(
        PROJECT_ROOT,
        game_state_build_id=game_state_build_id,
        universe_id=SHA_B,
        capture_sha256=capture_sha256,
    )

    assert authorization.registration_head_sha256 == amendment_head
    assert source_manifest_bundle_sha256 == canonical_sha256(
        {
            "schema": "nfl_x13_source_manifest_bundle_v1",
            "experiment_id": "X-13",
            "plan_id": X13_BATCH_SPEC.plan_id,
            "game_state_build_id": game_state_build_id,
            "universe_id": SHA_B,
            "capture_sha256": capture_sha256,
        }
    )
    assert authorization.analysis_lock_sha256 == (
        expected_analysis_lock_sha256(
            registration_head_sha256=amendment_head,
            game_state_build_id=game_state_build_id,
            universe_id=SHA_B,
            capture_sha256=capture_sha256,
        )
    )
    assert authorization.analysis_lock_sha256 != (
        expected_analysis_lock_sha256(
            registration_head_sha256=SHA_A,
            game_state_build_id=game_state_build_id,
            universe_id=SHA_B,
            capture_sha256=capture_sha256,
        )
    )

    source_lock = next(
        lock
        for lock in card["registration_locks"]
        if lock["id"] == "source_manifest_bundle"
    )
    source_lock["evidence_ref"] = capture_sha256
    with pytest.raises(
        X13PipelineError,
        match="source_manifest_bundle",
    ):
        load_x13_pipeline_authorization(
            PROJECT_ROOT,
            game_state_build_id=game_state_build_id,
            universe_id=SHA_B,
            capture_sha256=capture_sha256,
        )
    source_lock["evidence_ref"] = source_manifest_bundle_sha256

    effective_input = card["preregistered_inputs"][
        "preliminary_source_time_only"
    ]
    effective_input["code_sha256"] = SHA_A
    with pytest.raises(
        X13PipelineError,
        match="preregistered_inputs",
    ):
        load_x13_pipeline_authorization(
            PROJECT_ROOT,
            game_state_build_id=game_state_build_id,
            universe_id=SHA_B,
            capture_sha256=capture_sha256,
        )
    effective_input["code_sha256"] = expected_input["code_sha256"]

    del card["registration_head_sha256"]
    with pytest.raises(
        X13PipelineError,
        match="registration_head_sha256",
    ):
        load_x13_pipeline_authorization(
            PROJECT_ROOT,
            game_state_build_id=game_state_build_id,
            universe_id=SHA_B,
            capture_sha256=capture_sha256,
        )


def test_x13_code_bundle_rejects_noncanonical_or_symlink_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prediction_market.sports.nfl_x13_pipeline as pipeline_module

    from prediction_market.sports.nfl_x13_pipeline import (
        X13PipelineError,
        expected_x13_evaluation_code_bundle_sha256,
    )

    root = tmp_path / "program"
    root.mkdir()
    for invalid_path in (
        "/absolute/evaluation.py",
        "../outside.py",
        "src/../evaluation.py",
        "src//evaluation.py",
    ):
        monkeypatch.setattr(
            pipeline_module,
            "X13_EVALUATION_CODE_PATHS",
            (invalid_path,),
        )
        with pytest.raises(X13PipelineError, match="canonical relative POSIX"):
            expected_x13_evaluation_code_bundle_sha256(root)

    actual = root / "actual"
    actual.mkdir()
    (actual / "evaluation.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "linked").symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(
        pipeline_module,
        "X13_EVALUATION_CODE_PATHS",
        ("linked/evaluation.py",),
    )
    with pytest.raises(X13PipelineError, match="symlink"):
        expected_x13_evaluation_code_bundle_sha256(root)

    root_alias = tmp_path / "program-alias"
    root_alias.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(
        pipeline_module,
        "X13_EVALUATION_CODE_PATHS",
        ("actual/evaluation.py",),
    )
    with pytest.raises(X13PipelineError, match="symlink"):
        expected_x13_evaluation_code_bundle_sha256(root_alias)


def test_public_pipeline_entrypoints_cannot_inject_authorization() -> None:
    from prediction_market.sports.nfl_x13_pipeline import (
        execute_x13_pipeline,
        run_x13_full_batch,
    )

    assert "authorization" not in inspect.signature(
        execute_x13_pipeline
    ).parameters
    assert "authorization" not in inspect.signature(
        run_x13_full_batch
    ).parameters


def test_polymarket_json_numbers_preserve_exact_decimal_values() -> None:
    import prediction_market.sports.nfl_x13_pipeline as pipeline_module

    payload = pipeline_module._strict_json(
        b'{"price":0.8399999997,"size":4.48297618e2,"count":2}',
        context="Polymarket numeric trade fixture",
        exact_decimal_numbers=True,
    )

    assert payload == {
        "price": Decimal("0.8399999997"),
        "size": Decimal("448.297618"),
        "count": 2,
    }
    assert type(payload["count"]) is int
    default_payload = pipeline_module._strict_json(
        b'{"price":0.5}',
        context="non-trade JSON fixture",
    )
    assert type(default_payload["price"]) is float
    with pytest.raises(pipeline_module.X13PipelineError, match="non-finite"):
        pipeline_module._strict_json(
            b'{"price":NaN}',
            context="invalid Polymarket trade fixture",
            exact_decimal_numbers=True,
        )


def test_association_preview_is_market_covering_stratified_and_deterministic(
) -> None:
    import prediction_market.sports.nfl_x13_pipeline as pipeline_module

    rows = [
        {
            "episode_id": f"episode-{index}",
            "episode_type": "score" if index % 2 else "turnover",
            "logical_market_id": market_id,
            "outcome": "YES",
            "delay_scenario_seconds": index % 2,
            "horizon_seconds": 10 if index % 3 else 30,
            "validity_status": "OBSERVED",
        }
        for index, market_id in enumerate(
            ("market-a", "market-a", "market-b", "market-b", "market-c", "market-c")
        )
    ]

    first = pipeline_module._AssociationPreviewSampler(limit=4)
    second = pipeline_module._AssociationPreviewSampler(limit=4)
    for row in rows:
        first.add(row)
        second.add(dict(row))
    first.add(dict(rows[0]))
    second.add(dict(rows[0]))

    selected = first.finalize()

    assert selected == second.finalize()
    assert len(selected) == 4
    assert {row["logical_market_id"] for row in selected} == {
        "market-a",
        "market-b",
        "market-c",
    }
    assert len(
        {
            (
                row["logical_market_id"],
                row["delay_scenario_seconds"],
                row["horizon_seconds"],
                row["episode_type"],
                row["validity_status"],
            )
            for row in selected
        }
    ) == 4


def test_exploration_readiness_counts_episode_units_but_refuses_inference(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    import prediction_market.sports.nfl_x13_pipeline as pipeline_module

    rows = []
    for index, (
        game_id,
        episode_id,
        venue,
        validity_status,
    ) in enumerate(
        (
            ("game-a", "episode-1", "kalshi", "OBSERVED"),
            ("game-a", "episode-1", "kalshi", "OBSERVED"),
            ("game-a", "episode-1", "polymarket", "OBSERVED"),
            ("game-b", "episode-2", "kalshi", "OBSERVED"),
            ("game-b", "episode-3", "kalshi", "NO_POST_TRADE"),
        )
    ):
        rows.append(
            {
                "game_id": game_id,
                "episode_id": episode_id,
                "episode_type": "touchdown",
                "contract_id": f"contract-{index}",
                "logical_market_id": f"market-{index}",
                "venue": venue,
                "family": "moneyline",
                "outcome": "YES",
                "source_time_start_utc": "2026-01-01T00:00:00Z",
                "source_time_end_utc": "2026-01-01T00:00:01Z",
                "delay_scenario_seconds": 0,
                "horizon_seconds": 10,
                "pre_event_actual_trade": "0.4",
                "first_post_event_trade": "0.5",
                "vwap": "0.5",
                "signed_price_change": "0.1",
                "maximum_excursion": "0.1",
                "net_change_60s": "0.1",
                "trade_count": 1,
                "volume": "1",
                "staleness_seconds": "0",
                "overshoot_candidate": False,
                "reversal_candidate": False,
                "two_venue_direction_consistency": "UNAVAILABLE",
                "order_ambiguous": False,
                "contaminated": False,
                "validity_status": validity_status,
            }
        )
    path = tmp_path / "associations.parquet"
    table = pa.Table.from_pylist(
        rows,
        schema=pipeline_module._association_arrow_schema(),
    )
    pq.write_table(table, path)

    report = pipeline_module._exploration_readiness_report(
        association_paths=(path,),
        expected_candidate_row_count=5,
        input_root_analysis_lock_sha256=SHA_A,
    )

    assert report["candidate_evidence"]["candidate_row_count"] == 5
    assert report["candidate_evidence"]["observed_row_count"] == 4
    assert report["candidate_evidence"]["unique_episode_count"] == 2
    assert report["candidate_evidence"]["episode_venue_unit_count"] == 3
    assert report["candidate_evidence"]["repeated_episode_venue_unit_count"] == 1
    assert report["candidate_support_envelope"][
        "nominal_support_threshold_met"
    ] is False
    assert report["candidate_support_envelope"]["gate_evaluated"] is False
    assert report["candidate_support_envelope"]["status"] == (
        "NOT_RUN_PRIMARY_PROJECTION_NOT_FROZEN"
    )
    assert {
        item["hypothesis_id"] for item in report["registered_whitelist"]
    } == {
        hypothesis.hypothesis_id
        for hypothesis in (
            pipeline_module.X13_REGISTERED_ANALYSIS_LOCK_V1.hypotheses
        )
    }
    assert all(
        item["status"].startswith("NOT_RUN")
        for item in report["registered_whitelist"]
    )
    assert report["source_binding"]["registered_analysis_spec_lock_id"] == (
        pipeline_module.X13_REGISTERED_ANALYSIS_LOCK_V1.lock_id
    )
    assert report["source_binding"]["input_root_analysis_lock_sha256"] == SHA_A
    assert len(
        report["source_binding"]["association_partition_sha256s"]
    ) == 1
    assert report["claim_boundary"]["tradeable_alpha"] is False


def test_end_to_end_pipeline_reopens_evidence_normalizes_and_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prediction_market.sports.nfl_x13_batch as batch_module
    import prediction_market.sports.nfl_x13_pipeline as pipeline_module

    from prediction_market.sports.nfl_x13_batch import verify_published_batch
    from prediction_market.sports.nfl_x13_pipeline import (
        X13PipelineAuthorizationV1,
        execute_x13_pipeline,
        expected_analysis_lock_sha256,
        run_x13_full_batch,
    )

    game_build = _fixture_game_build()
    game_state_path = publish_x13_game_state(
        game_build,
        output_root=tmp_path / "game-state",
    )
    initial_universe = _initial_universe()
    capture = _capture_fixture(
        initial_universe,
        tmp_path / "raw",
        monkeypatch,
    )
    universe = initial_universe
    universe_path = publish_x13_universe(
        universe,
        (tmp_path / "universe" / "x13-universe.json").resolve(),
    )
    monkeypatch.setattr(
        pipeline_module, "_FROZEN_UNIVERSE_ID", universe.universe_id
    )
    monkeypatch.setattr(
        pipeline_module, "_FROZEN_ASSOCIATION_COUNT", 20 * 6 * 6 * 4
    )
    monkeypatch.setattr(
        batch_module, "X13_FROZEN_UNIVERSE_ID", universe.universe_id
    )
    monkeypatch.setattr(
        batch_module,
        "X13_FROZEN_ASSOCIATION_ROWS_BY_GAME",
        {game_id: 6 * 6 * 4 for game_id in X13_GAME_IDS},
    )
    monkeypatch.setattr(
        batch_module,
        "X13_FROZEN_ASSOCIATION_ROW_COUNT",
        20 * 6 * 6 * 4,
    )
    fixture_inventory_hashes = {}
    for game in universe.games:
        normalized = {
            contract.contract_id: contract
            for contract in pipeline_module.normalize_x13_contracts(game)
        }
        fixture_inventory_hashes[game.game_id] = (
            batch_module.contract_inventory_sha256(
                [
                    pipeline_module._contract_payload(
                        contract,
                        normalized[contract.contract_id],
                        (),
                    )
                    for contract in game.contracts
                ]
            )
        )
    monkeypatch.setattr(
        batch_module,
        "X13_FROZEN_CONTRACT_INVENTORY_SHA256_BY_GAME",
        fixture_inventory_hashes,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_verify_game_source_manifests",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_verify_universe_manifest_hashes",
        lambda *args, **kwargs: (),
    )
    analysis_lock = expected_analysis_lock_sha256(
        registration_head_sha256=SHA_A,
        game_state_build_id=str(game_build.manifest["build_id"]),
        universe_id=universe.universe_id,
        capture_sha256=capture.pointer.capture_sha256,
    )
    authorization = X13PipelineAuthorizationV1(
        experiment_id="X-13",
        scope="preliminary_source_time_only",
        registration_head_sha256=SHA_A,
        required_lock_ids=(
            "analysis_whitelist",
            "contract_universe",
            "game_sample_and_bindings",
            "h_split_approval",
            "source_manifest_bundle",
            "source_time_interval_semantics",
        ),
        resolved_lock_ids=(
            "analysis_whitelist",
            "contract_universe",
            "game_sample_and_bindings",
            "h_split_approval",
            "source_manifest_bundle",
            "source_time_interval_semantics",
        ),
        analysis_lock_sha256=analysis_lock,
        license_operational_use=(
            ("I-018", "APPROVED"),
            ("O-001", "RESEARCH_ONLY"),
            ("O-003", "RESEARCH_ONLY"),
            ("O-009", "RESEARCH_ONLY"),
        ),
    )
    authorization_loads: list[tuple[Path, str, str, str]] = []

    def load_test_authorization(
        program_root: str | Path,
        *,
        game_state_build_id: str,
        universe_id: str,
        capture_sha256: str,
    ) -> X13PipelineAuthorizationV1:
        authorization_loads.append(
            (
                Path(program_root).resolve(),
                game_state_build_id,
                universe_id,
                capture_sha256,
            )
        )
        return authorization

    monkeypatch.setattr(
        pipeline_module,
        "load_x13_pipeline_authorization",
        load_test_authorization,
    )

    result = execute_x13_pipeline(
        game_state_build=game_build,
        game_state_publication=game_state_path,
        universe=universe,
        universe_artifact=universe_path,
        capture=capture,
        program_root=PROJECT_ROOT,
        raw_store_root=tmp_path / "raw",
        output_root=tmp_path / "batch",
    )

    verified = verify_published_batch(result.batch_path)
    assert verified["game_count"] == 20
    auxiliary = result.batch_manifest["auxiliary_artifacts"]
    association_tables = [
        row
        for row in auxiliary
        if row["role"] == "association_candidate_table"
    ]
    assert association_tables
    assert sum(row["row_count"] for row in association_tables) == (
        result.runtime["association_count"]
    )
    assert all(
        row["media_type"] == "application/vnd.apache.parquet"
        and row["relative_path"].startswith("data/associations/")
        for row in association_tables
    )
    assert {
        row["role"] for row in auxiliary
    } >= {
        "association_cardinality_report",
        "association_runtime_rss_report",
        "contract_semantic_overlap_audit",
        "cross_game_mve_inventory",
        "validation_report",
        "lineage_report",
        "robinhood_provenance_inventory",
    }
    assert tuple(payload["game_id"] for payload in result.game_payloads) == (
        X13_GAME_IDS
    )
    first = result.game_payloads[0]
    assert first["audit"] == {
        "replay": "PASS",
        "personnel": "PASS_RESEARCH_ONLY",
        "layer_g": "PASS",
        "layer_m": "PASS",
        "layer_a": "PASS_SOURCE_TIME_ONLY",
        "manifest_hashes_verified": True,
        "unresolved": [
            "Team I terms review remains NOT_GREEN_OPEN; research-only"
        ],
    }
    assert {
        (row["venue"], row["provenance"], row["render_semantics"])
        for row in first["observations"]
    } >= {
        ("polymarket", "observed", "solid"),
        ("polymarket", "derived_complement", "dashed"),
        ("kalshi", "observed", "solid"),
    }
    assert all(
        row["bid"] is None and row["ask"] is None
        for row in first["observations"]
        if row["venue"] == "polymarket"
    )
    assert any(
        row["kind"] == "kalshi_1m_candle_bbo"
        and row["price"] is None
        and row["bid"] is not None
        and row["ask"] is not None
        and row["yes_bid_ohlc"]
        == {
            "open": "0.54",
            "high": "0.56",
            "low": "0.53",
            "close": "0.55",
        }
        and row["yes_ask_ohlc"]
        == {
            "open": "0.56",
            "high": "0.58",
            "low": "0.55",
            "close": "0.57",
        }
        and row["open_interest"] == "20"
        for row in first["observations"]
    )
    observation_id_fields = {
        "venue",
        "raw_market_id",
        "logical_market_id",
        "outcome",
        "kind",
        "source_start",
        "source_end",
        "source_end_inclusive",
        "price",
        "size",
        "bid",
        "ask",
        "provenance",
        "native_id",
        "derived_from_observation_id",
        "native_source_time",
        "is_block_trade",
        "taker_side",
        "taker_outcome_side",
        "yes_price_dollars",
        "no_price_dollars",
        "primary_path_eligible",
        "yes_bid_ohlc",
        "yes_ask_ohlc",
        "volume",
        "open_interest",
    }
    assert all(
        set(row)
        == observation_id_fields | {"observation_id", "render_semantics"}
        and row["observation_id"]
        == canonical_sha256(
            {
                field: row[field]
                for field in observation_id_fields
            }
        )
        for row in first["observations"]
    )
    assert first["market_observation_audit"]["block_trade_count"] == 1
    assert all(
        not row["primary_path_eligible"]
        for row in first["observations"]
        if row["is_block_trade"] is True
        or row["provenance"] == "derived_complement"
    )
    tie_game = next(
        payload
        for payload in result.game_payloads
        if payload["game_id"] == "2025_04_GB_DAL"
    )
    tie_kalshi_trades = [
        row
        for row in tie_game["observations"]
        if row["venue"] == "kalshi"
        and row["kind"] == "trade"
    ]
    assert {row["outcome"] for row in tie_kalshi_trades} == {"GB", "DAL"}
    assert all(
        row["provenance"] == "observed" for row in tie_kalshi_trades
    )
    tie_kalshi_winner = next(
        row
        for row in tie_game["contracts"]
        if row["logical_market_id"] == "kalshi:2025_04_GB_DAL:winner"
    )
    assert tie_kalshi_winner["outcome_coverage"] == {
        "GB": "available",
        "DAL": "available",
    }
    assert first["associations"]
    assert (
        first["association_coverage"]["actual_full_rows"]
        == first["association_coverage"]["expected_full_rows"]
    )
    assert first["association_coverage"]["presentation_filter"] == (
        "actual_market_evidence_and_ambiguous_or_contaminated_status"
    )
    assert first["association_coverage"]["presentation_row_count"] == len(
        first["associations"]
    )
    assert first["association_coverage"]["presentation_row_limit"] == 5_000
    assert (
        first["association_coverage"]["presentation_eligible_count"]
        == first["association_coverage"]["presentation_row_count"]
        + first["association_coverage"]["presentation_omitted_count"]
    )
    assert (
        first["association_coverage"]["presentation_row_count"] <= 5_000
    )
    assert all(
        row["pre_event_actual_trade"] is not None
        or row["first_post_event_trade"] is not None
        or row["trade_count"] > 0
        for row in first["associations"]
    )
    association = first["associations"][0]
    assert set(association) >= {
        "episode_id",
        "episode_type",
        "source_time_start_utc",
        "source_time_end_utc",
        "logical_market_id",
        "signed_price_change",
        "maximum_excursion",
        "net_change_60s",
        "validity_status",
        "delay_scenario_seconds",
        "horizon_seconds",
    }
    assert first["lineage"]["analysis_lock"] == analysis_lock
    assert result.runtime["game_count"] == 20
    assert result.runtime["compute_worker_count"] == 2
    assert result.runtime["storage_pilot_row_count"] > 0
    assert result.runtime["running_disk_gate_flush_count"] > 0
    assert result.runtime["association_presentation_count"] == sum(
        len(payload["associations"]) for payload in result.game_payloads
    )
    cardinality_report = json.loads(
        (
            result.batch_path
            / "reports/association-cardinality-report-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert cardinality_report["compute_worker_count"] == 2
    assert cardinality_report["storage_pilot"]["row_count"] > 0
    assert cardinality_report["storage_pilot"]["row_limit"] == 65_536
    assert (
        cardinality_report["storage_pilot"]["row_count"]
        <= cardinality_report["storage_pilot"]["row_limit"]
    )
    assert cardinality_report["running_disk_gate_flush_count"] > 0

    reloaded_capture = load_x13_capture(
        capture.pointer_path,
        plans=universe.capture_plans,
        universe_id=universe.universe_id,
        raw_store_root=tmp_path / "raw",
        program_root=PROJECT_ROOT,
    )
    rebuilt_runtime = {
        **dict(game_build.manifest["runtime"]),
        "elapsed_ns": int(game_build.manifest["runtime"]["elapsed_ns"]) + 1,
        "peak_rss_bytes": (
            int(game_build.manifest["runtime"]["peak_rss_bytes"]) + 1
        ),
    }
    rebuilt_game_state = replace(
        game_build,
        manifest=MappingProxyType(
            {
                **dict(game_build.manifest),
                "runtime": rebuilt_runtime,
            }
        ),
    )
    assert rebuilt_game_state.manifest["runtime"] != (
        game_build.manifest["runtime"]
    )
    assert rebuilt_game_state.manifest["build_id"] == (
        game_build.manifest["build_id"]
    )
    second = run_x13_full_batch(
        game_state_build=rebuilt_game_state,
        game_state_publication=game_state_path,
        universe_artifact=universe_path,
        capture=reloaded_capture,
        program_root=PROJECT_ROOT,
        raw_store_root=tmp_path / "raw",
        output_root=tmp_path / "batch-second",
    )
    first_tables = [
        {
            key: row[key]
            for key in (
                "relative_path",
                "row_count",
                "schema_fingerprint",
                "sha256",
                "byte_length",
            )
        }
        for row in result.batch_manifest["auxiliary_artifacts"]
        if row["role"] == "association_candidate_table"
    ]
    second_tables = [
        {
            key: row[key]
            for key in (
                "relative_path",
                "row_count",
                "schema_fingerprint",
                "sha256",
                "byte_length",
            )
        }
        for row in second.batch_manifest["auxiliary_artifacts"]
        if row["role"] == "association_candidate_table"
    ]
    assert first_tables == second_tables
    assert authorization_loads == [
        (
            PROJECT_ROOT.resolve(),
            str(game_build.manifest["build_id"]),
            universe.universe_id,
            capture.pointer.capture_sha256,
        ),
        (
            PROJECT_ROOT.resolve(),
            str(game_build.manifest["build_id"]),
            universe.universe_id,
            capture.pointer.capture_sha256,
        ),
    ]


def test_association_storage_preflight_preserves_five_gib_reserve() -> None:
    from prediction_market.sports.nfl_x13_pipeline import (
        X13PipelineError,
        preflight_x13_association_storage,
    )

    with pytest.raises(X13PipelineError, match="5 GiB"):
        preflight_x13_association_storage(
            available_free_bytes=6 * GIB,
            already_written_bytes=1,
            remaining_association_rows=10,
            empirical_bytes_per_row=1024,
        )


def test_association_running_disk_gate_checks_before_each_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prediction_market.sports.nfl_x13_pipeline as pipeline_module

    called = False

    def operation() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        pipeline_module.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=6 * GIB - 1),
    )
    gate = pipeline_module._AssociationWriteDiskGate(tmp_path)

    with pytest.raises(
        pipeline_module.X13PipelineError,
        match="before partition flush",
    ):
        gate.write(operation)

    assert not called
    assert gate.flush_count == 0


def test_canonical_universe_loader_reconstructs_exact_typed_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prediction_market.sports.nfl_x13_pipeline as pipeline_module

    from prediction_market.sports.nfl_x13_pipeline import (
        load_x13_universe_artifact,
    )

    universe = _initial_universe()
    artifact = publish_x13_universe(
        universe,
        (tmp_path / "universe" / "x13-universe.json").resolve(),
    )
    monkeypatch.setattr(
        pipeline_module, "_FROZEN_UNIVERSE_ID", universe.universe_id
    )
    loaded = load_x13_universe_artifact(artifact)
    assert loaded == universe
    assert loaded.canonical_payload == universe.canonical_payload


def test_universe_verifier_rejects_nonfrozen_contract_universe() -> None:
    from prediction_market.sports.nfl_x13_pipeline import (
        X13PipelineError,
        verify_x13_universe_artifact,
    )

    with pytest.raises(X13PipelineError, match="frozen X-13"):
        verify_x13_universe_artifact(
            _initial_universe(), Path("/does/not/matter")
        )


def test_game_state_source_manifests_must_reopen_from_raw_store(
    tmp_path: Path,
) -> None:
    from prediction_market.sports.nfl_x13_pipeline import (
        X13PipelineError,
        _verify_game_source_manifests,
    )

    with pytest.raises(X13PipelineError, match="cannot be resolved"):
        _verify_game_source_manifests(
            _fixture_game_build(),
            program_root=PROJECT_ROOT,
            raw_store_root=tmp_path,
        )


def test_real_frozen_winner_contracts_use_canonical_team_orientation() -> None:
    from prediction_market.sports.nfl_x13_association import (
        _semantic_orientation_key,
    )
    from prediction_market.sports.nfl_x13_pipeline import (
        _canonical_winner_outcome,
        load_x13_universe_artifact,
        normalize_x13_contracts,
    )

    artifact = (
        PROJECT_ROOT
        / "artifacts/market-observation/nfl/x13/universe"
        / "sha256-01526155f3436a752cd6765d5b6a88086ab71badac84e82518a526cc8a7098d2"
        / "x13-universe-plan-v1.json"
    )
    universe = load_x13_universe_artifact(artifact)
    for game, binding in zip(
        universe.games, X13_GAME_BINDINGS, strict=True
    ):
        contracts = normalize_x13_contracts(game)
        winners = [
            contract
            for contract in contracts
            if contract.logical_market_id.endswith(":winner")
        ]
        assert len(winners) == 2
        assert {
            contract.outcomes for contract in winners
        } == {(binding.away_team, binding.home_team)}
        assert {
            contract.subject for contract in winners
        } == {game.game_id}
        for team in (binding.away_team, binding.home_team):
            assert len(
                {
                    _semantic_orientation_key(contract, team)
                    for contract in winners
                }
            ) == 1

    dal_det = next(
        game
        for game in universe.games
        if game.game_id == "2025_14_DAL_DET"
    )
    poly_winner = next(
        contract
        for contract in dal_det.contracts
        if contract.logical_market_id
        == "polymarket:2025_14_DAL_DET:winner"
    )
    assert _canonical_winner_outcome(
        dal_det, poly_winner, "Cowboys"
    ) == "DAL"
    assert _canonical_winner_outcome(
        dal_det, poly_winner, "Lions"
    ) == "DET"


def test_real_frozen_structured_game_totals_use_canonical_orientation() -> None:
    from collections import defaultdict

    from prediction_market.sports.nfl_x13_association import (
        _semantic_orientation_key,
    )
    from prediction_market.sports.nfl_x13_pipeline import (
        _canonical_market_outcome,
        load_x13_universe_artifact,
        normalize_x13_contracts,
    )

    artifact = (
        PROJECT_ROOT
        / "artifacts/market-observation/nfl/x13/universe"
        / "sha256-01526155f3436a752cd6765d5b6a88086ab71badac84e82518a526cc8a7098d2"
        / "x13-universe-plan-v1.json"
    )
    universe = load_x13_universe_artifact(artifact)
    semantic_venues: dict[tuple[object, ...], set[str]] = defaultdict(set)
    total_contract_count = 0
    for game in universe.games:
        normalized_by_id = {
            contract.contract_id: contract
            for contract in normalize_x13_contracts(game)
        }
        for native in game.contracts:
            if not (
                native.family == "total"
                and native.period == "game"
                and native.measure == "combined_points"
                and native.comparator == "gt"
                and native.line is not None
            ):
                continue
            total_contract_count += 1
            normalized = normalized_by_id[native.contract_id]
            assert normalized.subject == game.game_id
            assert normalized.outcomes == ("Over", "Under")
            for native_outcome in native.outcomes:
                canonical_outcome = {
                    "yes": "Over",
                    "over": "Over",
                    "no": "Under",
                    "under": "Under",
                }[native_outcome.casefold()]
                assert (
                    _canonical_market_outcome(
                        game,
                        native,
                        native_outcome,
                    )
                    == canonical_outcome
                )
            for outcome in normalized.outcomes:
                semantic_venues[
                    (
                        game.game_id,
                        *_semantic_orientation_key(normalized, outcome),
                    )
                ].add(normalized.venue)

    matched_orientations = {
        key
        for key, venues in semantic_venues.items()
        if venues == {"polymarket", "kalshi"}
    }
    assert total_contract_count == 434
    assert len(matched_orientations) == 114


def test_real_contract_semantic_overlap_audit_proves_winner_and_total() -> None:
    from prediction_market.sports.nfl_x13_pipeline import (
        _contract_semantic_overlap_audit,
        load_x13_universe_artifact,
        normalize_x13_contracts,
    )

    artifact = (
        PROJECT_ROOT
        / "artifacts/market-observation/nfl/x13/universe"
        / "sha256-01526155f3436a752cd6765d5b6a88086ab71badac84e82518a526cc8a7098d2"
        / "x13-universe-plan-v1.json"
    )
    universe = load_x13_universe_artifact(artifact)
    prepared = tuple(
        SimpleNamespace(
            universe_game=game,
            contracts=normalize_x13_contracts(game),
        )
        for game in universe.games
    )

    audit = _contract_semantic_overlap_audit(prepared)

    assert audit["status"] == "PASS"
    assert audit["winner_game_count"] == 20
    assert audit["winner_orientation_count"] == 40
    assert audit["structured_total_contract_count"] == 434
    assert audit["matched_total_line_count"] == 57
    assert audit["matched_total_orientation_count"] == 114
    assert audit["actual_cross_venue_orientation_count"] == 154
    assert audit["approved_cross_venue_orientation_count"] == 154
    assert audit["unapproved_cross_venue_orientation_count"] == 0
    assert len(audit["winner_records"]) == 20
    assert len(audit["matched_total_records"]) == 57
    assert audit["unresolved_family_policy"] == (
        "team, player, period, spread, and composite propositions remain "
        "SINGLE_VENUE unless structured native fields prove the same subject, "
        "period, comparator, line, and orientation; titles are not parsed"
    )


def test_real_cross_venue_overlap_is_exactly_the_audit_approved_set() -> None:
    from collections import defaultdict

    from prediction_market.sports.nfl_x13_association import (
        _semantic_orientation_key,
    )
    from prediction_market.sports.nfl_x13_pipeline import (
        load_x13_universe_artifact,
        normalize_x13_contracts,
    )

    artifact = (
        PROJECT_ROOT
        / "artifacts/market-observation/nfl/x13/universe"
        / "sha256-01526155f3436a752cd6765d5b6a88086ab71badac84e82518a526cc8a7098d2"
        / "x13-universe-plan-v1.json"
    )
    universe = load_x13_universe_artifact(artifact)
    semantic_venues: dict[tuple[object, ...], set[str]] = defaultdict(set)
    approved_keys: set[tuple[object, ...]] = set()
    kincaid_subjects: set[str] = set()
    for game in universe.games:
        for contract in normalize_x13_contracts(game):
            is_winner = (
                contract.family == "moneyline"
                and contract.period == "game"
                and contract.measure == "winner"
            )
            is_structured_total = (
                contract.family == "total"
                and contract.period == "game"
                and contract.measure == "combined_points"
                and contract.comparator == "gt"
                and contract.line is not None
                and contract.outcomes == ("Over", "Under")
            )
            if not (is_winner or is_structured_total):
                assert contract.subject.startswith("SINGLE_VENUE:")
            if (
                game.game_id == "2025_20_BUF_DEN"
                and contract.family == "any_td"
                and "Kincaid" in next(
                    native.subject
                    for native in game.contracts
                    if native.contract_id == contract.contract_id
                )
            ):
                kincaid_subjects.add(contract.subject)
            for outcome in contract.outcomes:
                key = (
                    game.game_id,
                    *_semantic_orientation_key(contract, outcome),
                )
                semantic_venues[key].add(contract.venue)
                if is_winner or is_structured_total:
                    approved_keys.add(key)

    actual_overlap = {
        key
        for key, venues in semantic_venues.items()
        if venues == {"polymarket", "kalshi"}
    }
    approved_overlap = {
        key
        for key in approved_keys
        if semantic_venues[key] == {"polymarket", "kalshi"}
    }
    assert len(kincaid_subjects) == 2
    assert actual_overlap == approved_overlap
    assert len(actual_overlap) == 154


def test_pipeline_rejects_tampered_pointer_before_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prediction_market.sports.nfl_x13_pipeline import (
        X13PipelineError,
        verify_x13_capture_evidence,
    )

    universe = _initial_universe()
    capture = _capture_fixture(universe, tmp_path / "raw", monkeypatch)
    capture.pointer_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(X13PipelineError, match="capture pointer"):
        verify_x13_capture_evidence(
            capture,
            universe=universe,
            program_root=PROJECT_ROOT,
            raw_store_root=tmp_path / "raw",
        )


def test_pipeline_rejects_tampered_capture_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prediction_market.sports.nfl_x13_pipeline import (
        X13PipelineError,
        verify_x13_capture_evidence,
    )

    universe = _initial_universe()
    capture = _capture_fixture(universe, tmp_path / "raw", monkeypatch)
    capture.receipt_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(X13PipelineError, match="capture receipt"):
        verify_x13_capture_evidence(
            capture,
            universe=universe,
            program_root=PROJECT_ROOT,
            raw_store_root=tmp_path / "raw",
        )


def test_pipeline_rejects_unknown_family_and_cross_game_analysis() -> None:
    from prediction_market.sports.nfl_x13_pipeline import (
        X13PipelineError,
        normalize_x13_contracts,
    )

    game = _initial_universe().games[0]
    unknown = replace(game.contracts[0], family="mystery")
    with pytest.raises(X13PipelineError, match="family"):
        normalize_x13_contracts(replace(game, contracts=(unknown,)))

    cross_game = replace(
        game.contracts[0],
        dependency_game_ids=(X13_GAME_IDS[0], X13_GAME_IDS[1]),
        analysis_eligible=True,
    )
    with pytest.raises(X13PipelineError, match="cross-game"):
        normalize_x13_contracts(replace(game, contracts=(cross_game,)))


def test_pipeline_requires_exact_twenty_games() -> None:
    from prediction_market.sports.nfl_x13_pipeline import (
        X13PipelineError,
        verify_x13_universe_artifact,
    )

    universe = _initial_universe()
    shortened = replace(
        universe,
        games=universe.games[:-1],
        capture_plans=universe.capture_plans[:-1],
    )
    with pytest.raises(X13PipelineError, match="exact frozen 20"):
        verify_x13_universe_artifact(shortened, Path("/does/not/matter"))


def test_analysis_lock_binds_all_three_immutable_input_roots() -> None:
    from prediction_market.sports.nfl_x13_pipeline import (
        expected_analysis_lock_sha256,
    )

    base = expected_analysis_lock_sha256(
        registration_head_sha256=SHA_A,
        game_state_build_id=SHA_B,
        universe_id=SHA_C,
        capture_sha256=f"sha256:{'d' * 64}",
    )
    changed = expected_analysis_lock_sha256(
        registration_head_sha256=SHA_A,
        game_state_build_id=SHA_B,
        universe_id=SHA_C,
        capture_sha256=f"sha256:{'e' * 64}",
    )
    assert base.startswith("sha256:")
    assert base != changed
