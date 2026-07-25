from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

import prediction_market.sports.nfl_x13_batch as batch_module
from prediction_market.sports.nfl_x13_batch import (
    AuxiliaryArtifactV1,
    X13_ASSOCIATION_SCHEMA_FINGERPRINT,
    X13BatchPublishError,
    association_arrow_schema,
    build_batch_manifest,
    canonical_payload_hash,
    contract_inventory_sha256,
    observation_payload_sha256,
    publish_x13_batch,
    render_game_html,
    verify_published_batch,
)
from prediction_market.sports.nfl_x13_spec import (
    X13_BATCH_SPEC,
    X13_GAME_BINDINGS,
    X13_GAME_IDS,
)


_BINDING_BY_GAME_ID = {
    binding.native_game_id: binding for binding in X13_GAME_BINDINGS
}
_REQUIRED_REPORTS = {
    "association_cardinality_report": (
        "reports/association-cardinality-report-v1.json",
        "nfl_x13_association_cardinality_report_v1",
    ),
    "association_runtime_rss_report": (
        "reports/runtime-rss-v1.json",
        "nfl_x13_association_runtime_rss_report_v1",
    ),
    "robinhood_provenance_inventory": (
        "inventory/robinhood-provenance-inventory-v1.json",
        "nfl_x13_robinhood_provenance_inventory_v1",
    ),
    "contract_semantic_overlap_audit": (
        "reports/contract-semantic-overlap-audit-v1.json",
        "nfl_x13_contract_semantic_overlap_audit_v1",
    ),
    "cross_game_mve_inventory": (
        "inventory/cross-game-mve-inventory-v1.json",
        "nfl_x13_cross_game_mve_inventory_v1",
    ),
    "validation_report": (
        "reports/validation-report-v1.json",
        "nfl_x13_validation_report_v1",
    ),
    "lineage_report": (
        "reports/lineage-report-v1.json",
        "nfl_x13_lineage_report_v1",
    ),
}


def _game_payload(game_id: str) -> dict[str, object]:
    binding = _BINDING_BY_GAME_ID[game_id]
    away = binding.away_team
    home = binding.home_team
    polymarket_logical_id = f"polymarket:{game_id}:winner"
    kalshi_logical_id = f"kalshi:{game_id}:winner"
    payload = {
        "schema": "nfl_x13_game_bundle_v1",
        "experiment_id": "X-13",
        "status": "PRELIMINARY_SOURCE_TIME_ONLY",
        "game_id": game_id,
        "teams": {
            "away": away,
            "home": home,
            "away_color": "#2563eb",
            "home_color": "#ef4444",
        },
        "final_score": {
            "away": binding.away_score,
            "home": binding.home_score,
        },
        "events": [
            {
                "game_id": game_id,
                "play_id": "1",
                "order_sequence": 1,
                "source_time_start_utc": "2025-09-07T17:00:00Z",
                "source_time_end_utc": "2025-09-07T17:00:01Z",
                "quarter": 1,
                "clock_seconds_remaining": 900,
                "home_team": home,
                "away_team": away,
                "possession_team": away,
                "offense_team": away,
                "defense_team": home,
                "down": 1,
                "distance": 10,
                "yardline_100": 75,
                "direction": "right",
                "field_coordinate_0_100": 25,
                "field_orientation_semantics": (
                    "canonical_team_end_orientation_not_tracking_coordinate"
                ),
                "play_family": "run",
                "formation": "SHOTGUN",
                "offense_personnel": "11",
                "defense_personnel": "nickel",
                "offense_names": [f"{away} Player {index}" for index in range(11)],
                "defense_names": [f"{home} Player {index}" for index in range(11)],
                "between_play_roster_change": None,
                "timeout": False,
                "penalty_status": "none",
                "touchdown": False,
                "turnover": False,
                "review": False,
                "review_result": "none",
                "no_play": False,
                "game_end": True,
                "home_score": binding.home_score,
                "away_score": binding.away_score,
                "description": "Run for five yards.",
            }
        ],
        "episodes": [
            {
                "episode_id": f"{game_id}:1",
                "episode_type": "run",
                "play_ids": ["1"],
                "source_time_start_utc": "2025-09-07T17:00:00Z",
                "source_time_end_utc": "2025-09-07T17:00:01Z",
                "contaminated": False,
            }
        ],
        "personnel": {"coverage": "research_only", "complete_11v11_rows": 1},
        "stat_ledger": [{"play_id": "1", "away_rushing_yards": 5}],
        "contracts": [
            {
                "contract_id": (
                    f"polymarket:{binding.polymarket_condition_id}"
                ),
                "condition_id": binding.polymarket_condition_id,
                "venue_market_id": binding.polymarket_market_id,
                "venue": "polymarket",
                "logical_market_id": polymarket_logical_id,
                "family": "moneyline",
                "kind": "primitive",
                "dependency_game_ids": [game_id],
                "analysis_eligible": True,
                "outcomes": [away, home],
                "outcome_coverage": {
                    away: "available",
                    home: "unavailable_no_trades",
                },
            },
            {
                "contract_id": f"kalshi:{binding.kalshi_event_ticker}",
                "venue": "kalshi",
                "venue_market_id": binding.kalshi_event_ticker,
                "raw_contract_ids": [
                    binding.kalshi_away_ticker,
                    binding.kalshi_home_ticker,
                ],
                "logical_market_id": kalshi_logical_id,
                "family": "moneyline",
                "kind": "primitive",
                "dependency_game_ids": [game_id],
                "analysis_eligible": True,
                "outcomes": [away, home],
                "outcome_coverage": {
                    away: "available",
                    home: "unavailable_no_trades",
                },
            },
        ],
        "observations": [
            {
                "observation_id": f"sha256:{'1' * 64}",
                "venue": "polymarket",
                "raw_market_id": (
                    f"polymarket:{binding.polymarket_condition_id}"
                ),
                "logical_market_id": polymarket_logical_id,
                "outcome": away,
                "kind": "trade",
                "source_start": "2025-09-07T17:00:02Z",
                "source_end": "2025-09-07T17:00:03Z",
                "source_end_inclusive": False,
                "price": "0.55",
                "size": "1",
                "bid": None,
                "ask": None,
                "provenance": "observed",
                "render_semantics": "solid",
                "derived_from_observation_id": None,
                "native_id": "poly-trade-1",
                "native_source_time": "2025-09-07T17:00:02Z",
                "is_block_trade": None,
                "taker_side": None,
                "taker_outcome_side": None,
                "yes_price_dollars": None,
                "no_price_dollars": None,
                "primary_path_eligible": True,
                "yes_bid_ohlc": None,
                "yes_ask_ohlc": None,
                "volume": None,
                "open_interest": None,
            },
            {
                "observation_id": f"sha256:{'2' * 64}",
                "venue": "polymarket",
                "raw_market_id": (
                    f"polymarket:{binding.polymarket_condition_id}"
                ),
                "logical_market_id": polymarket_logical_id,
                "outcome": home,
                "kind": "trade",
                "source_start": "2025-09-07T17:00:02Z",
                "source_end": "2025-09-07T17:00:03Z",
                "source_end_inclusive": False,
                "price": "0.45",
                "size": "1",
                "bid": None,
                "ask": None,
                "provenance": "derived_complement",
                "render_semantics": "dashed",
                "derived_from_observation_id": f"sha256:{'1' * 64}",
                "native_id": "poly-trade-1",
                "native_source_time": "2025-09-07T17:00:02Z",
                "is_block_trade": None,
                "taker_side": None,
                "taker_outcome_side": None,
                "yes_price_dollars": None,
                "no_price_dollars": None,
                "primary_path_eligible": False,
                "yes_bid_ohlc": None,
                "yes_ask_ohlc": None,
                "volume": None,
                "open_interest": None,
            },
            {
                "observation_id": f"sha256:{'3' * 64}",
                "venue": "kalshi",
                "raw_market_id": binding.kalshi_away_ticker,
                "logical_market_id": kalshi_logical_id,
                "outcome": away,
                "kind": "trade",
                "source_start": "2025-09-07T17:00:02Z",
                "source_end": "2025-09-07T17:00:03Z",
                "source_end_inclusive": False,
                "price": "0.56",
                "size": "1",
                "bid": None,
                "ask": None,
                "provenance": "observed",
                "render_semantics": "solid",
                "derived_from_observation_id": None,
                "native_id": "kalshi-trade-1",
                "native_source_time": "2025-09-07T17:00:02.123456Z",
                "is_block_trade": False,
                "taker_side": "yes",
                "taker_outcome_side": "yes",
                "yes_price_dollars": "0.56",
                "no_price_dollars": "0.44",
                "primary_path_eligible": True,
                "yes_bid_ohlc": None,
                "yes_ask_ohlc": None,
                "volume": None,
                "open_interest": None,
            },
            {
                "observation_id": f"sha256:{'4' * 64}",
                "venue": "kalshi",
                "raw_market_id": binding.kalshi_away_ticker,
                "logical_market_id": kalshi_logical_id,
                "outcome": home,
                "kind": "trade",
                "source_start": "2025-09-07T17:00:02Z",
                "source_end": "2025-09-07T17:00:03Z",
                "source_end_inclusive": False,
                "price": "0.44",
                "size": "1",
                "bid": None,
                "ask": None,
                "provenance": "derived_complement",
                "render_semantics": "dashed",
                "derived_from_observation_id": f"sha256:{'3' * 64}",
                "native_id": "kalshi-trade-1",
                "native_source_time": "2025-09-07T17:00:02.123456Z",
                "is_block_trade": False,
                "taker_side": "yes",
                "taker_outcome_side": "yes",
                "yes_price_dollars": "0.56",
                "no_price_dollars": "0.44",
                "primary_path_eligible": False,
                "yes_bid_ohlc": None,
                "yes_ask_ohlc": None,
                "volume": None,
                "open_interest": None,
            },
        ],
        "associations": [],
        "association_coverage": {
            "expected_full_rows": 144,
            "actual_full_rows": 144,
            "presentation_row_count": 0,
            "presentation_eligible_count": 0,
            "presentation_omitted_count": 0,
            "presentation_row_limit": 5_000,
            "presentation_selection_order": (
                "first_rows_in_canonical_association_stream_order"
            ),
            "presentation_filter": (
                "actual_market_evidence_and_ambiguous_or_contaminated_status"
            ),
            "full_table_format": "partitioned_parquet",
            "full_table_partitions": [
                {
                    "relative_path": (
                        "data/associations/"
                        f"game_id={game_id}/"
                        "venue=polymarket/"
                        "family=moneyline/"
                        "part-00000.parquet"
                    ),
                    "row_count": 144,
                    "sha256": f"sha256:{'f' * 64}",
                    "schema_fingerprint": (
                        X13_ASSOCIATION_SCHEMA_FINGERPRINT
                    ),
                }
            ],
        },
        "market_coverage": {
            "polymarket_historical_bbo": "unavailable",
            "kalshi_historical_bbo": "one_minute_candle_only",
        },
        "audit": {
            "replay": "PASS",
            "personnel": "PASS_RESEARCH_ONLY",
            "layer_g": "PASS",
            "layer_m": "PASS",
            "layer_a": "PASS_SOURCE_TIME_ONLY",
            "manifest_hashes_verified": True,
            "unresolved": [],
        },
        "lineage": {
            "raw_manifest_hashes": [f"sha256:{'a' * 64}"],
            "builder_version": "nfl-x13-v1",
            "analysis_lock": f"sha256:{'b' * 64}",
        },
    }
    observations = payload["observations"]
    observations[0]["observation_id"] = observation_payload_sha256(
        observations[0]
    )
    observations[1]["derived_from_observation_id"] = observations[0][
        "observation_id"
    ]
    observations[1]["observation_id"] = observation_payload_sha256(
        observations[1]
    )
    observations[2]["observation_id"] = observation_payload_sha256(
        observations[2]
    )
    observations[3]["derived_from_observation_id"] = observations[2][
        "observation_id"
    ]
    observations[3]["observation_id"] = observation_payload_sha256(
        observations[3]
    )
    return payload


def _payloads() -> list[dict[str, object]]:
    return [_game_payload(game_id) for game_id in X13_GAME_IDS]


def _candle_observation(
    *,
    raw_market_id: str,
    logical_market_id: str,
    outcome: str,
    bid: str,
    ask: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "observation_id": f"sha256:{'0' * 64}",
        "venue": "kalshi",
        "raw_market_id": raw_market_id,
        "logical_market_id": logical_market_id,
        "outcome": outcome,
        "kind": "kalshi_1m_candle_bbo",
        "source_start": "2025-09-07T17:00:00Z",
        "source_end": "2025-09-07T17:01:00Z",
        "source_end_inclusive": False,
        "price": None,
        "size": "10",
        "bid": bid,
        "ask": ask,
        "provenance": "observed",
        "render_semantics": "solid",
        "derived_from_observation_id": None,
        "native_id": f"{raw_market_id}:1757264460:1m",
        "native_source_time": "2025-09-07T17:01:00Z",
        "is_block_trade": None,
        "taker_side": None,
        "taker_outcome_side": None,
        "yes_price_dollars": None,
        "no_price_dollars": None,
        "primary_path_eligible": True,
        "yes_bid_ohlc": {
            "open": bid,
            "high": bid,
            "low": bid,
            "close": bid,
        },
        "yes_ask_ohlc": {
            "open": ask,
            "high": ask,
            "low": ask,
            "close": ask,
        },
        "volume": "10",
        "open_interest": "5",
    }
    row["observation_id"] = observation_payload_sha256(row)
    return row


def _association_descriptor_mappings(
    payloads: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    values = []
    for payload in payloads:
        partition = payload["association_coverage"][
            "full_table_partitions"
        ][0]
        values.append(
            {
                "schema": "nfl_x13_auxiliary_artifact_v1",
                "relative_path": partition["relative_path"],
                "role": "association_candidate_table",
                "media_type": "application/vnd.apache.parquet",
                "schema_fingerprint": partition["schema_fingerprint"],
                "row_count": partition["row_count"],
                "sha256": partition["sha256"],
                "byte_length": 1,
            }
        )
    for role, (relative_path, schema_name) in _REQUIRED_REPORTS.items():
        values.append(
            {
                "schema": "nfl_x13_auxiliary_artifact_v1",
                "relative_path": relative_path,
                "role": role,
                "media_type": "application/json",
                "schema_fingerprint": canonical_payload_hash(
                    {
                        "schema": schema_name,
                        "version": "canonical-json-v1",
                    }
                ),
                "row_count": 1,
                "sha256": f"sha256:{'e' * 64}",
                "byte_length": 1,
            }
        )
    return tuple(values)


def _build_manifest(
    payloads: list[dict[str, object]],
    *,
    plan_id: str = X13_BATCH_SPEC.plan_id,
    builder_version: str = "nfl-x13-v1",
) -> dict[str, object]:
    return build_batch_manifest(
        payloads,
        plan_id=plan_id,
        builder_version=builder_version,
        auxiliary_artifacts=_association_descriptor_mappings(payloads),
    )


def _association_auxiliary_artifacts(
    tmp_path: Path,
    payloads: list[dict[str, object]],
) -> tuple[AuxiliaryArtifactV1, ...]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = association_arrow_schema()
    values: list[AuxiliaryArtifactV1] = []
    source_root = tmp_path / "association-fixtures"
    source_root.mkdir(parents=True, exist_ok=True)
    for payload in payloads:
        game_id = str(payload["game_id"])
        row_count = int(
            payload["association_coverage"]["expected_full_rows"]
        )
        data: dict[str, list[object]] = {}
        for field in schema:
            if field.nullable:
                value: object = None
            elif pa.types.is_string(field.type):
                value = (
                    game_id
                    if field.name == "game_id"
                    else "polymarket"
                    if field.name == "venue"
                    else "moneyline"
                    if field.name == "family"
                    else "fixture"
                )
            elif pa.types.is_boolean(field.type):
                value = False
            else:
                value = 0
            data[field.name] = [value] * row_count
        table = pa.Table.from_pydict(data, schema=schema)
        source = source_root / f"{game_id}.parquet"
        pq.write_table(
            table,
            source,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        partition = payload["association_coverage"][
            "full_table_partitions"
        ][0]
        partition["sha256"] = f"sha256:{digest}"
        values.append(
            AuxiliaryArtifactV1(
                relative_path=partition["relative_path"],
                source_path=source,
                role="association_candidate_table",
                media_type="application/vnd.apache.parquet",
                schema_fingerprint=X13_ASSOCIATION_SCHEMA_FINGERPRINT,
                row_count=row_count,
                sha256=f"sha256:{digest}",
                byte_length=source.stat().st_size,
            )
        )
    return tuple(values)


def _required_json_auxiliary_artifacts(
    tmp_path: Path,
) -> tuple[AuxiliaryArtifactV1, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    values: list[AuxiliaryArtifactV1] = []
    for role, (relative_path, schema_name) in _REQUIRED_REPORTS.items():
        source = tmp_path / f"{role}.json"
        raw = (
            json.dumps(
                {"schema": schema_name},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        source.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        values.append(
            AuxiliaryArtifactV1(
                relative_path=relative_path,
                source_path=source,
                role=role,
                media_type="application/json",
                schema_fingerprint=canonical_payload_hash(
                    {
                        "schema": schema_name,
                        "version": "canonical-json-v1",
                    }
                ),
                row_count=1,
                sha256=f"sha256:{digest}",
                byte_length=len(raw),
            )
        )
    return tuple(values)


def _publish_batch(
    payloads: list[dict[str, object]],
    *,
    output_root: Path,
) -> Path:
    auxiliary = _association_auxiliary_artifacts(
        output_root.parent / f"{output_root.name}-sources",
        payloads,
    )
    reports = _required_json_auxiliary_artifacts(
        output_root.parent / f"{output_root.name}-report-sources"
    )
    return publish_x13_batch(
        payloads,
        output_root=output_root,
        plan_id=X13_BATCH_SPEC.plan_id,
        builder_version="nfl-x13-v1",
        auxiliary_artifacts=(*auxiliary, *reports),
    )


@pytest.fixture(autouse=True)
def _freeze_bounded_fixture_universe(monkeypatch) -> None:
    payloads = _payloads()
    monkeypatch.setattr(
        batch_module,
        "X13_FROZEN_ASSOCIATION_ROWS_BY_GAME",
        {game_id: 144 for game_id in X13_GAME_IDS},
    )
    monkeypatch.setattr(
        batch_module,
        "X13_FROZEN_ASSOCIATION_ROW_COUNT",
        144 * len(X13_GAME_IDS),
    )
    monkeypatch.setattr(
        batch_module,
        "X13_FROZEN_CONTRACT_INVENTORY_SHA256_BY_GAME",
        {
            str(payload["game_id"]): contract_inventory_sha256(
                payload["contracts"]
            )
            for payload in payloads
        },
    )


def _rewrite_checksum_ledger(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    contents = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(root).as_posix()}\n"
        for path in files
    )
    (root / "checksums.sha256").write_text(contents, encoding="ascii")


def test_render_is_offline_generic_and_shows_both_outcomes() -> None:
    payload = _game_payload("2025_01_BAL_BUF")
    payload["associations"] = [
        {
            "episode_id": "2025_01_BAL_BUF:1",
            "episode_type": "touchdown",
            "logical_market_id": "polymarket:2025_01_BAL_BUF:winner",
            "outcome": "BAL",
            "source_time_start_utc": "2025-09-07T17:00:00Z",
            "source_time_end_utc": "2025-09-07T17:00:01Z",
            "delay_scenario_seconds": 0,
            "horizon_seconds": 10,
            "pre_event_actual_trade": "0.45",
            "first_post_event_trade": "0.55",
            "vwap": "0.54",
            "signed_price_change": "0.10",
            "maximum_excursion": "0.12",
            "net_change_60s": "0.08",
            "trade_count": 3,
            "volume": "14",
            "staleness_seconds": "1",
            "overshoot_candidate": True,
            "reversal_candidate": False,
            "two_venue_direction_consistency": "CONSISTENT",
            "order_ambiguous": False,
            "contaminated": False,
            "validity_status": "OBSERVED",
        }
    ]
    payload["association_coverage"]["presentation_row_count"] = 1
    payload["association_coverage"]["presentation_eligible_count"] = 1
    html = render_game_html(payload)

    assert "<!doctype html>" in html.lower()
    assert "BAL" in html and "BUF" in html
    assert "橄榄球场" in html
    assert "历史 bid/ask unavailable" in html
    assert "derived / non-observed / non-executable" in html
    assert "source-time" in html
    assert "Delay scenario" in html
    assert "Horizon" in html
    assert "VWAP" in html
    assert "source_time_start_utc" in html
    assert "overshoot_candidate" in html
    assert "new Date(t).toISOString()" in html
    assert "PRELIMINARY_SOURCE_TIME_ONLY" in html
    assert "https://" not in html
    assert "http://" not in html
    assert "<script src=" not in html
    assert "<link " not in html
    assert 'data-provenance="observed"' in html
    assert 'data-provenance="derived_complement"' in html
    assert "const observedRows=" in html
    assert "const derivedRows=" in html


def test_render_draws_real_kalshi_candle_bid_ask_without_fabricating_price(
) -> None:
    payload = _game_payload("2025_01_BAL_BUF")
    kalshi = next(
        contract
        for contract in payload["contracts"]
        if contract["venue"] == "kalshi"
    )
    kalshi["outcome_coverage"] = {
        "BAL": "unavailable_no_trades",
        "BUF": "unavailable_no_trades",
    }
    payload["observations"] = [
        observation
        for observation in payload["observations"]
        if observation["venue"] != "kalshi"
    ] + [
        _candle_observation(
            raw_market_id=_BINDING_BY_GAME_ID[
                "2025_01_BAL_BUF"
            ].kalshi_away_ticker,
            logical_market_id="kalshi:2025_01_BAL_BUF:winner",
            outcome="BAL",
            bid="0.54",
            ask="0.57",
        ),
        _candle_observation(
            raw_market_id=_BINDING_BY_GAME_ID[
                "2025_01_BAL_BUF"
            ].kalshi_home_ticker,
            logical_market_id="kalshi:2025_01_BAL_BUF:winner",
            outcome="BUF",
            bid="0.43",
            ask="0.46",
        ),
    ]

    rendered = render_game_html(payload)

    assert "BBO bid" in rendered
    assert "BBO ask" in rendered
    assert "o.bid!=null&&o.ask!=null" in rendered
    assert 'x2="1160" y2="315" stroke="#47627f"/>' in rendered
    assert 'class="bbo-point"' in rendered
    assert 'class="candle-bbo"' in rendered
    assert 'id="market-candle-detail"' in rendered
    assert "Bid O/H/L/C" in rendered
    assert "volume / open interest" in rendered


def test_manifest_requires_exact_frozen_twenty_and_green_source_layers() -> None:
    payloads = _payloads()
    manifest = _build_manifest(payloads)

    assert manifest["game_ids"] == list(X13_GAME_IDS)
    assert manifest["game_count"] == 20
    assert manifest["publication_gate"] == "PASS"
    assert manifest["batch_id"].startswith("sha256:")
    assert manifest["experiment_id"] == "X-13"

    with pytest.raises(X13BatchPublishError, match="exact frozen 20"):
        _build_manifest(payloads[:-1])

    failed = deepcopy(payloads)
    failed[0]["audit"]["layer_m"] = "FAIL"
    with pytest.raises(X13BatchPublishError, match="Layer G/M"):
        _build_manifest(failed)


def test_cross_game_contract_cannot_enter_single_game_analysis() -> None:
    payloads = _payloads()
    payloads[0]["contracts"][0]["dependency_game_ids"] = [
        X13_GAME_IDS[0],
        X13_GAME_IDS[1],
    ]
    payloads[0]["contracts"][0]["analysis_eligible"] = True

    with pytest.raises(X13BatchPublishError, match="cross-game"):
        _build_manifest(payloads)


def test_atomic_publish_writes_twenty_html_json_index_and_verifies(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    published = _publish_batch(payloads, output_root=tmp_path)

    assert published.name.startswith("sha256-")
    assert (published / "index.html").is_file()
    assert (published / "batch_manifest.json").is_file()
    assert (published / "checksums.sha256").is_file()
    assert len(list((published / "games").glob("*.html"))) == 20
    assert len(list((published / "games").glob("*.json"))) == 20
    verified = verify_published_batch(published)
    assert verified["verified_file_count"] == 70
    assert verified["game_count"] == 20

    second = _publish_batch(payloads, output_root=tmp_path)
    assert second == published

    event_file = published / "games" / f"{X13_GAME_IDS[0]}.json"
    event_file.write_bytes(event_file.read_bytes() + b" ")
    with pytest.raises(X13BatchPublishError, match="checksum"):
        verify_published_batch(published)


def test_auxiliary_artifacts_are_content_bound_copied_and_verified(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    association_artifacts = _association_auxiliary_artifacts(
        tmp_path / "required-sources",
        payloads,
    )
    report_artifacts = _required_json_auxiliary_artifacts(
        tmp_path / "required-report-sources"
    )
    artifact = next(
        value
        for value in report_artifacts
        if value.role == "validation_report"
    )

    published = publish_x13_batch(
        payloads,
        output_root=tmp_path / "published",
        plan_id=X13_BATCH_SPEC.plan_id,
        builder_version="nfl-x13-v1",
        auxiliary_artifacts=(*association_artifacts, *report_artifacts),
    )

    copied = published / artifact.relative_path
    assert copied.read_bytes() == artifact.source_path.read_bytes()
    manifest = json.loads(
        (published / "batch_manifest.json").read_text("utf-8")
    )
    descriptor = next(
        value
        for value in manifest["auxiliary_artifacts"]
        if value["relative_path"] == artifact.relative_path
    )
    assert descriptor == artifact.descriptor
    assert verify_published_batch(published)["verified_file_count"] == 70

    copied.write_bytes(copied.read_bytes() + b"tamper")
    _rewrite_checksum_ledger(published)
    with pytest.raises(X13BatchPublishError, match="auxiliary artifact"):
        verify_published_batch(published)


def test_auxiliary_artifact_source_and_path_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "report.json"
    source.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(X13BatchPublishError, match="safe relative path"):
        AuxiliaryArtifactV1(
            relative_path="../report.json",
            source_path=source,
            role="validation",
            media_type="application/json",
            schema_fingerprint=f"sha256:{'d' * 64}",
            row_count=None,
            sha256=f"sha256:{digest}",
            byte_length=2,
        )

    payloads = _payloads()
    required = list(
        _association_auxiliary_artifacts(
            tmp_path / "required-sources",
            payloads,
        )
    )
    reports = list(
        _required_json_auxiliary_artifacts(
            tmp_path / "required-report-sources"
        )
    )
    validation_index = next(
        index
        for index, artifact in enumerate(reports)
        if artifact.role == "validation_report"
    )
    reports[validation_index] = replace(
        reports[validation_index],
        source_path=source,
        sha256=f"sha256:{'e' * 64}",
        byte_length=2,
    )
    with pytest.raises(X13BatchPublishError, match="source hash"):
        publish_x13_batch(
            payloads,
            output_root=tmp_path / "published",
            plan_id=X13_BATCH_SPEC.plan_id,
            builder_version="nfl-x13-v1",
            auxiliary_artifacts=(*required, *reports),
        )


def test_invalid_association_parquet_is_rejected_before_atomic_publish(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    required = list(
        _association_auxiliary_artifacts(
            tmp_path / "association-sources",
            payloads,
        )
    )
    corrupt = tmp_path / "not-parquet.bin"
    corrupt.write_bytes(b"not parquet\n")
    digest = hashlib.sha256(corrupt.read_bytes()).hexdigest()
    first = required[0]
    payloads[0]["association_coverage"]["full_table_partitions"][0][
        "sha256"
    ] = f"sha256:{digest}"
    required[0] = replace(
        first,
        source_path=corrupt,
        sha256=f"sha256:{digest}",
        byte_length=corrupt.stat().st_size,
    )
    output_root = tmp_path / "published"

    with pytest.raises(X13BatchPublishError, match="not valid Parquet"):
        publish_x13_batch(
            payloads,
            output_root=output_root,
            plan_id=X13_BATCH_SPEC.plan_id,
            builder_version="nfl-x13-v1",
            auxiliary_artifacts=(
                *required,
                *_required_json_auxiliary_artifacts(
                    tmp_path / "required-report-sources"
                ),
            ),
        )

    assert output_root.is_dir()
    assert list(output_root.iterdir()) == []


def test_canonical_hash_is_order_stable_but_not_content_blind() -> None:
    left = {"b": [2, 1], "a": {"x": "0.5"}}
    right = {"a": {"x": "0.5"}, "b": [2, 1]}
    changed = {"a": {"x": "0.5"}, "b": [1, 2]}

    assert canonical_payload_hash(left) == canonical_payload_hash(right)
    assert canonical_payload_hash(left) != canonical_payload_hash(changed)


def test_manifest_rejects_nonfrozen_plan_id() -> None:
    with pytest.raises(X13BatchPublishError, match="frozen X-13 plan"):
        _build_manifest(
            _payloads(),
            plan_id=f"sha256:{'c' * 64}",
        )


def test_manifest_rejects_binary_float_anywhere_in_payload() -> None:
    payloads = _payloads()
    payloads[0]["events"][0]["binary_float_probe"] = 0.1

    with pytest.raises(X13BatchPublishError, match="binary float"):
        _build_manifest(payloads)


def test_manifest_accepts_spec_decimal_price_without_binary_float() -> None:
    payloads = _payloads()
    payloads[0]["observations"][0]["price"] = Decimal("0.55")

    manifest = _build_manifest(payloads)

    assert manifest["publication_gate"] == "PASS"


def test_manifest_rejects_observation_material_tamper_without_new_identity() -> None:
    payloads = _payloads()
    payloads[0]["observations"][0]["size"] = "999"

    with pytest.raises(
        X13BatchPublishError,
        match="observation_id does not match canonical material",
    ):
        _build_manifest(payloads)


def test_manifest_rejects_team_or_final_score_outside_frozen_binding() -> None:
    wrong_team = _payloads()
    wrong_team[0]["teams"]["away"] = "XXX"
    with pytest.raises(X13BatchPublishError, match="frozen team binding"):
        _build_manifest(wrong_team)

    wrong_score = _payloads()
    wrong_score[0]["final_score"]["away"] = 999
    with pytest.raises(X13BatchPublishError, match="frozen final score"):
        _build_manifest(wrong_score)


def test_manifest_rejects_foreign_event_identity_and_nonterminal_score() -> None:
    foreign = _payloads()
    foreign[0]["events"][0]["game_id"] = X13_GAME_IDS[1]
    with pytest.raises(X13BatchPublishError, match="event identity"):
        _build_manifest(foreign)

    nonterminal = _payloads()
    nonterminal[0]["events"][-1]["game_end"] = False
    with pytest.raises(X13BatchPublishError, match="terminal event"):
        _build_manifest(nonterminal)


def test_manifest_rejects_field_orientation_inconsistent_with_possession() -> None:
    payloads = _payloads()
    payloads[0]["events"][0]["direction"] = "left"
    payloads[0]["events"][0]["field_coordinate_0_100"] = 75

    with pytest.raises(X13BatchPublishError, match="field orientation"):
        _build_manifest(payloads)


def test_manifest_requires_frozen_winner_inventory_from_both_venues() -> None:
    payloads = _payloads()
    payloads[0]["contracts"] = [
        contract
        for contract in payloads[0]["contracts"]
        if contract["venue"] != "polymarket"
    ]
    payloads[0]["observations"] = [
        observation
        for observation in payloads[0]["observations"]
        if observation["venue"] != "polymarket"
    ]

    with pytest.raises(X13BatchPublishError, match="winner inventory"):
        _build_manifest(payloads)


def test_manifest_requires_each_observed_winner_outcome_path() -> None:
    payloads = _payloads()
    logical_market_id = payloads[0]["contracts"][0]["logical_market_id"]
    home = payloads[0]["teams"]["home"]
    payloads[0]["contracts"][0]["outcome_coverage"][home] = "available"
    payloads[0]["observations"] = [
        observation
        for observation in payloads[0]["observations"]
        if not (
            observation["logical_market_id"] == logical_market_id
            and observation["outcome"] == home
        )
    ]

    with pytest.raises(X13BatchPublishError, match="outcome path"):
        _build_manifest(payloads)


def test_real_no_trade_outcome_is_explicitly_unavailable_without_fake_path() -> None:
    payload = _game_payload(X13_GAME_IDS[0])
    logical_market_id = payload["contracts"][0]["logical_market_id"]
    for outcome in payload["contracts"][0]["outcomes"]:
        payload["contracts"][0]["outcome_coverage"][
            outcome
        ] = "unavailable_no_trades"
    payload["observations"] = [
        observation
        for observation in payload["observations"]
        if observation["logical_market_id"] != logical_market_id
    ]

    html = render_game_html(payload)

    assert "coverage unavailable · no fabricated path" in html


def test_derived_complement_does_not_upgrade_observed_outcome_coverage() -> None:
    payloads = _payloads()
    contract = payloads[0]["contracts"][0]
    logical_market_id = contract["logical_market_id"]
    home = payloads[0]["teams"]["home"]
    contract["outcome_coverage"][home] = "unavailable_no_trades"
    payloads[0]["observations"] = [
        observation
        for observation in payloads[0]["observations"]
        if not (
            observation["logical_market_id"] == logical_market_id
            and observation["outcome"] == home
            and observation["provenance"] == "observed"
        )
    ]

    manifest = _build_manifest(payloads)

    assert manifest["publication_gate"] == "PASS"


def test_derived_complement_is_exactly_bound_to_its_observed_source() -> None:
    payloads = _payloads()
    payloads[0]["observations"][1]["price"] = "0.46"
    payloads[0]["observations"][1][
        "observation_id"
    ] = observation_payload_sha256(payloads[0]["observations"][1])

    with pytest.raises(X13BatchPublishError, match="exact complement"):
        _build_manifest(payloads)

    payloads = _payloads()
    payloads[0]["observations"][1][
        "derived_from_observation_id"
    ] = f"sha256:{'9' * 64}"
    payloads[0]["observations"][1][
        "observation_id"
    ] = observation_payload_sha256(payloads[0]["observations"][1])
    with pytest.raises(
        X13BatchPublishError,
        match="source is missing or non-observed",
    ):
        _build_manifest(payloads)


def test_no_trade_coverage_can_retain_real_candle_bbo_without_price_path() -> None:
    payloads = _payloads()
    payload = payloads[0]
    contract = payload["contracts"][1]
    logical_market_id = contract["logical_market_id"]
    for outcome in contract["outcomes"]:
        contract["outcome_coverage"][outcome] = "unavailable_no_trades"
    payload["observations"] = [
        observation
        for observation in payload["observations"]
        if observation["logical_market_id"] != logical_market_id
    ]
    payload["observations"].append(
        _candle_observation(
            raw_market_id=_BINDING_BY_GAME_ID[
                payload["game_id"]
            ].kalshi_away_ticker,
            logical_market_id=logical_market_id,
            outcome=payload["teams"]["away"],
            bid="0.40",
            ask="0.60",
        )
    )

    manifest = _build_manifest(payloads)

    assert manifest["publication_gate"] == "PASS"


def test_renderer_consumes_canonical_field_coordinate_directly() -> None:
    html = render_game_html(_game_payload(X13_GAME_IDS[0]))

    assert "e.field_coordinate_0_100" in html
    assert "rawCoordinate==null?NaN" in html
    assert "if(hasField)" in html
    assert "field coordinate / direction unavailable" in html
    assert "100-y100" not in html
    assert "真实方向" not in html
    assert "非 tracking 坐标" in html


def test_verifier_rejects_game_json_tamper_with_rewritten_checksum(
    tmp_path: Path,
) -> None:
    published = _publish_batch(_payloads(), output_root=tmp_path)
    game_path = published / "games" / f"{X13_GAME_IDS[0]}.json"
    game = json.loads(game_path.read_text("utf-8"))
    game["final_score"]["away"] = 999
    game_path.write_text(
        json.dumps(
            game,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _rewrite_checksum_ledger(published)

    with pytest.raises(X13BatchPublishError):
        verify_published_batch(published)
    with pytest.raises(X13BatchPublishError):
        _publish_batch(_payloads(), output_root=tmp_path)


def test_verifier_rejects_html_tamper_with_rewritten_checksum(
    tmp_path: Path,
) -> None:
    published = _publish_batch(_payloads(), output_root=tmp_path)
    html_path = published / "games" / f"{X13_GAME_IDS[0]}.html"
    html_path.write_text(
        html_path.read_text("utf-8").replace(
            "standalone offline artifact",
            "tampered artifact",
        ),
        encoding="utf-8",
    )
    _rewrite_checksum_ledger(published)

    with pytest.raises(X13BatchPublishError, match="HTML"):
        verify_published_batch(published)


def test_verifier_recomputes_batch_id_from_manifest_identity(
    tmp_path: Path,
) -> None:
    published = _publish_batch(_payloads(), output_root=tmp_path)
    manifest_path = published / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["builder_version"] = "tampered-builder"
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _rewrite_checksum_ledger(published)

    with pytest.raises(X13BatchPublishError, match="batch_id"):
        verify_published_batch(published)
