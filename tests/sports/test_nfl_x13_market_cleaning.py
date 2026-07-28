from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from prediction_market.sports.nfl_x13_market import (
    derive_complement,
    normalize_kalshi_candle_bbo,
    normalize_kalshi_trade,
    normalize_polymarket_trade,
)


def _poly_trade():
    return normalize_polymarket_trade(
        {
            "transactionHash": "0xtrade",
            "timestamp": 1_765_000_000,
            "price": "0.61",
            "size": "4",
        },
        raw_market_id="0xcondition",
        logical_market_id="polymarket:0xcondition:winner",
        outcome="JAX",
    )


def _kalshi_candle():
    return normalize_kalshi_candle_bbo(
        {
            "end_period_ts": 1_765_000_040,
            "yes_bid": {
                "open": "0.54",
                "high": "0.56",
                "low": "0.53",
                "close": "0.55",
            },
            "yes_ask": {
                "open": "0.57",
                "high": "0.59",
                "low": "0.56",
                "close": "0.58",
            },
            "volume": "10",
            "open_interest": "25",
        },
        ticker="KXNFLGAME-25NOV09JACHOU-JAC",
        outcome="yes",
        logical_market_id="kalshi:2025_10_JAX_HOU:winner",
    )


def _kalshi_trade():
    return normalize_kalshi_trade(
        {
            "trade_id": "kalshi-trade-1",
            "created_time": "2025-12-05T01:00:00.250000Z",
            "is_block_trade": False,
            "taker_side": "yes",
            "taker_outcome_side": "yes",
            "yes_price_dollars": "0.61",
            "no_price_dollars": "0.39",
            "count_fp": "4",
        },
        ticker="KXNFLGAME-25NOV09JACHOU-JAC",
        outcome="yes",
        logical_market_id="kalshi:2025_10_JAX_HOU:winner",
    )


def test_v2_sha256_requires_exact_lowercase_canonical_text() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        X13MarketRecleanError,
        _require_v2_sha256,
    )

    assert _require_v2_sha256(_sha("a"), "digest") == _sha("a")
    with pytest.raises(X13MarketRecleanError, match="canonical"):
        _require_v2_sha256("sha256:" + "A" * 64, "digest")


def test_cleaning_rows_keep_trade_candle_and_derived_semantics_distinct() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        MarketCleaningInputV1,
        clean_market_rows_v1,
    )

    trade = _poly_trade()
    candle = _kalshi_candle()
    complement = derive_complement(trade, outcome="HOU")
    audit = clean_market_rows_v1(
        (
            MarketCleaningInputV1(
                game_id="2025_10_JAX_HOU",
                observation=trade,
                contract_id="polymarket:0xcondition",
                event_id="2025_10_JAX_HOU",
                token_id="token-jax",
                ticker=None,
                proposition="Jacksonville wins",
                orientation="JAX",
                raw_row={
                    "transactionHash": "0xtrade",
                    "timestamp": 1_765_000_000,
                    "price": "0.61",
                    "size": "4",
                },
            ),
            MarketCleaningInputV1(
                game_id="2025_10_JAX_HOU",
                observation=candle,
                contract_id="kalshi:KXNFLGAME-25NOV09JACHOU-JAC",
                event_id="KXNFLGAME-25NOV09JACHOU",
                token_id=None,
                ticker="KXNFLGAME-25NOV09JACHOU-JAC",
                proposition="Jacksonville wins",
                orientation="JAC",
                raw_row={
                    "end_period_ts": 1_765_000_040,
                    "yes_bid_close": Decimal("0.55"),
                    "yes_ask_close": Decimal("0.58"),
                },
            ),
            MarketCleaningInputV1(
                game_id="2025_10_JAX_HOU",
                observation=complement,
                contract_id="polymarket:0xcondition",
                event_id="2025_10_JAX_HOU",
                token_id="token-hou-derived",
                ticker=None,
                proposition="Houston wins",
                orientation="HOU",
                raw_row={
                    "derived_from_observation_id": trade.observation_id,
                },
            ),
        )
    )

    assert audit.status == "PASS"
    assert len(audit.cleaned_observations) == 3
    by_kind = {row.observation_kind: row for row in audit.rows}
    actual = by_kind["actual_trade"]
    assert actual.actual_observation is True
    assert actual.actual_trade is True
    assert actual.bbo_available is False
    assert actual.bbo_tick is False
    assert actual.executable is False
    candle_row = by_kind["kalshi_1m_candle"]
    assert candle_row.actual_observation is True
    assert candle_row.actual_trade is False
    assert candle_row.bbo_available is True
    assert candle_row.bbo_tick is False
    assert candle_row.executable is False
    assert candle_row.orientation == "JAX"
    assert candle_row.ticker == "KXNFLGAME-25NOV09JACHOU-JAC"
    derived = by_kind["derived_complement"]
    assert derived.actual_observation is False
    assert derived.actual_trade is False
    assert derived.bbo_available is False
    assert derived.executable is False
    assert derived.provenance == "derived_complement"
    assert derived.dedupe_decision == "KEEP"
    records = audit.to_records()
    assert records == [row.to_dict() for row in audit.rows]
    assert {record["game_id"] for record in records} == {
        "2025_10_JAX_HOU"
    }
    assert all(record["raw_fingerprint"].startswith("sha256:") for record in records)
    assert audit.to_dict()["status"] == "PASS"


def _cleaning_input(observation, raw_row):
    from prediction_market.sports.nfl_x13_market_cleaning import (
        MarketCleaningInputV1,
    )

    return MarketCleaningInputV1(
        game_id="2025_10_JAX_HOU",
        observation=observation,
        contract_id="polymarket:0xcondition",
        event_id="2025_10_JAX_HOU",
        token_id="token-jax",
        ticker=None,
        proposition="Jacksonville wins",
        orientation="JAX",
        raw_row=raw_row,
    )


def test_polymarket_native_id_dedupes_exact_rows_and_rejects_conflicts() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        clean_market_rows_v1,
    )

    trade = _poly_trade()
    raw = {
        "transactionHash": "0xtrade",
        "timestamp": 1_765_000_000,
        "price": "0.61",
        "size": "4",
    }
    exact = clean_market_rows_v1(
        (_cleaning_input(trade, raw), _cleaning_input(trade, raw))
    )

    assert exact.status == "PASS"
    assert [row.dedupe_decision for row in exact.rows] == [
        "KEEP",
        "DROP_EXACT_DUPLICATE",
    ]
    assert len(exact.cleaned_observations) == 1

    conflicting = replace(
        trade,
        observation_id="sha256:" + "a" * 64,
        price=Decimal("0.62"),
    )
    collision = clean_market_rows_v1(
        (
            _cleaning_input(trade, raw),
            _cleaning_input(
                conflicting,
                {**raw, "price": "0.62"},
            ),
        )
    )

    assert collision.status == "FAIL"
    assert collision.cleaned_observations == ()
    assert {row.dedupe_decision for row in collision.rows} == {
        "EXCLUDE_COLLISION"
    }
    assert {row.collision_reason for row in collision.rows} == {
        "conflicting_economics_for_semantic_identity"
    }


def test_polymarket_missing_native_id_uses_economics_and_raw_fingerprint() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        build_polymarket_cleaning_input_v1,
        clean_market_rows_v1,
    )

    raw = {
        "timestamp": 1_765_000_000,
        "price": "0.61",
        "size": "4",
        "side": "BUY",
    }

    def source(value):
        return build_polymarket_cleaning_input_v1(
            value,
            game_id="2025_10_JAX_HOU",
            raw_market_id="0xcondition",
            logical_market_id="polymarket:0xcondition:winner",
            outcome="JAX",
            contract_id="polymarket:0xcondition",
            event_id="2025_10_JAX_HOU",
            token_id="token-jax",
            proposition="Jacksonville wins",
            orientation="JAX",
        )

    missing_native = source(raw).observation
    exact = clean_market_rows_v1(
        (
            source(raw),
            source(raw),
        )
    )
    assert exact.status == "PASS"
    assert [row.dedupe_decision for row in exact.rows] == [
        "KEEP",
        "DROP_EXACT_DUPLICATE",
    ]
    assert exact.rows[0].native_id is None
    assert exact.cleaned_observations[0].native_id.startswith("fallback:")

    distinct_raw = clean_market_rows_v1(
        (
            source({**raw, "source_page": 1}),
            source({**raw, "source_page": 2}),
        )
    )
    assert distinct_raw.status == "PASS"
    assert [row.dedupe_decision for row in distinct_raw.rows] == [
        "KEEP",
        "KEEP",
    ]
    assert len({row.dedupe_key for row in distinct_raw.rows}) == 2
    assert len(distinct_raw.cleaned_observations) == 2

    conflicting = replace(
        missing_native,
        observation_id="sha256:" + "c" * 64,
        price=Decimal("0.62"),
    )
    collision = clean_market_rows_v1(
        (
            _cleaning_input(missing_native, raw),
            _cleaning_input(conflicting, {**raw, "price": "0.62"}),
        )
    )
    assert collision.status == "FAIL"
    assert {row.collision_reason for row in collision.rows} == {
        "conflicting_economics_for_semantic_identity"
    }


def test_kalshi_trade_id_is_stable_and_missing_identity_is_excluded() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        MarketCleaningInputV1,
        clean_market_rows_v1,
    )

    trade = _kalshi_trade()

    def source(observation):
        return MarketCleaningInputV1(
            game_id="2025_10_JAX_HOU",
            observation=observation,
            contract_id="kalshi:KXNFLGAME-25NOV09JACHOU-JAC",
            event_id="KXNFLGAME-25NOV09JACHOU",
            token_id=None,
            ticker="KXNFLGAME-25NOV09JACHOU-JAC",
            proposition="Jacksonville wins",
            orientation="JAC",
            raw_row={
                "trade_id": observation.native_id,
                "created_time": "2025-12-05T01:00:00.250000Z",
                "yes_price_dollars": str(observation.price),
                "count_fp": str(observation.size),
            },
        )

    exact = clean_market_rows_v1((source(trade), source(trade)))
    assert exact.status == "PASS"
    assert [row.dedupe_decision for row in exact.rows] == [
        "KEEP",
        "DROP_EXACT_DUPLICATE",
    ]
    assert exact.cleaned_observations[0].native_id == "kalshi-trade-1"
    assert exact.rows[0].orientation == "JAX"

    missing = clean_market_rows_v1(
        (
            source(
                replace(
                    trade,
                    observation_id="sha256:" + "d" * 64,
                    native_id="",
                )
            ),
        )
    )
    assert missing.status == "FAIL"
    assert missing.cleaned_observations == ()
    assert missing.rows[0].dedupe_decision == "EXCLUDE"
    assert missing.rows[0].unknown_reason == "missing_kalshi_trade_id"
    assert missing.rows[0].exclusion_reason == "missing_kalshi_trade_id"


def test_cleaning_exposes_unknown_identity_and_invalid_orientation_reasons() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        MarketCleaningInputV1,
        clean_market_rows_v1,
    )

    invalid = MarketCleaningInputV1(
        game_id="2025_10_JAX_HOU",
        observation=_poly_trade(),
        contract_id="polymarket:0xcondition",
        event_id="2025_10_JAX_HOU",
        token_id=None,
        ticker=None,
        proposition="Jacksonville wins",
        orientation=None,
        raw_row={"transactionHash": "0xtrade"},
    )
    audit = clean_market_rows_v1((invalid,))

    assert audit.status == "FAIL"
    assert audit.cleaned_observations == ()
    assert audit.rows[0].unknown_reason == "missing_polymarket_token_id"
    assert audit.rows[0].exclusion_reason == "invalid_orientation"


def test_cleaning_rejects_missing_or_blank_game_id() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        MarketCleaningInputV1,
    )

    kwargs = {
        "observation": _poly_trade(),
        "contract_id": "polymarket:0xcondition",
        "event_id": "2025_10_JAX_HOU",
        "token_id": "token-jax",
        "ticker": None,
        "proposition": "Jacksonville wins",
        "orientation": "JAX",
        "raw_row": {"transactionHash": "0xtrade"},
    }
    with pytest.raises(TypeError, match="game_id must be nonempty text"):
        MarketCleaningInputV1(game_id="", **kwargs)
    with pytest.raises(TypeError, match="game_id must be nonempty text"):
        MarketCleaningInputV1(game_id="   ", **kwargs)
    with pytest.raises(TypeError, match="game_id must be nonempty text"):
        MarketCleaningInputV1(game_id=None, **kwargs)


def test_cleaning_rejects_rows_from_more_than_one_game() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        MarketCleaningError,
        clean_market_rows_v1,
    )

    first = _cleaning_input(
        _poly_trade(),
        {"transactionHash": "0xtrade"},
    )
    second = replace(first, game_id="2025_14_DAL_DET")

    with pytest.raises(
        MarketCleaningError,
        match="exactly one game_id",
    ):
        clean_market_rows_v1((first, second))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_ARTIFACT = (
    PROJECT_ROOT
    / "artifacts/market-observation/nfl/x13/universe"
    / "sha256-01526155f3436a752cd6765d5b6a88086ab71badac84e82518a526cc8a7098d2"
    / "x13-universe-plan-v1.json"
)
CAPTURE_SHA256 = "sha256:" + "c" * 64


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _real_first_game():
    from prediction_market.sports.nfl_x13_pipeline import (
        load_x13_universe_artifact,
    )

    return load_x13_universe_artifact(UNIVERSE_ARTIFACT).games[0]


def _capture_receipt(
    *,
    manifest_sha256: str,
    object_sha256: str,
    request_sha256: str,
):
    from prediction_market.sports.nfl_x13_capture import (
        ImmutableRequestManifest,
    )

    return ImmutableRequestManifest(
        request_sha256=request_sha256,
        object_sha256=object_sha256,
        manifest_sha256=manifest_sha256,
        byte_length=1,
        object_path=Path("/verified/raw.json"),
        manifest_path=Path("/verified/raw.manifest.json"),
        response_received_at_utc=datetime(
            2026,
            7,
            25,
            tzinfo=timezone.utc,
        ),
    )


def _verified_document(
    *,
    game_id: str,
    resource: str,
    manifest_sha256: str,
    object_sha256: str,
    source_request: dict[str, object],
    source_cursor: str,
    payload: object,
):
    from prediction_market.sports.nfl_x13_pipeline import (
        VerifiedCaptureDocumentV1,
    )

    return VerifiedCaptureDocumentV1(
        game_id=game_id,
        resource=resource,
        manifest_sha256=manifest_sha256,
        object_sha256=object_sha256,
        byte_length=1,
        source_url="https://verified.example.invalid/offline",
        source_request=source_request,
        source_cursor=source_cursor,
        coverage="verified fixture coverage",
        dataset_id="DS-POLYMARKET-PUBLIC"
        if resource == "polymarket_trades"
        else "DS-KALSHI-HISTORICAL",
        license_ref="O-001"
        if resource == "polymarket_trades"
        else "O-003",
        license_status="pending",
        current_license_status="research_only",
        current_authority_snapshot_sha256=_sha("a"),
        current_catalog_registry_sha256=_sha("b"),
        current_data_license_registry_sha256=_sha("c"),
        current_dataset_registry_sha256=_sha("f"),
        current_experiment_registry_sha256=_sha("d"),
        payload=payload,
    )


def _polymarket_verified_fixture():
    from prediction_market.sports.nfl_x13_capture import (
        GameCaptureResult,
        PolymarketTerminalWindow,
    )

    game = _real_first_game()
    binding = game.polymarket_token_bindings[0]
    contract = next(
        value
        for value in game.contracts
        if value.condition_id == binding.condition_id
    )
    raw = {
        "asset": binding.token_ids[0],
        "conditionId": binding.condition_id,
        "outcome": binding.outcomes[0],
        "price": "0.61",
        "size": "4",
        "timestamp": game.source_window.start_ts,
        "transactionHash": "0xverified-poly-trade",
    }
    manifest_sha256 = _sha("1")
    object_sha256 = _sha("2")
    request_sha256 = _sha("3")
    source_cursor = (
        f"condition:{binding.condition_id};"
        f"start:{game.source_window.start_ts};"
        f"end:{game.source_window.end_ts};offset:0"
    )
    document = _verified_document(
        game_id=game.game_id,
        resource="polymarket_trades",
        manifest_sha256=manifest_sha256,
        object_sha256=object_sha256,
        source_request={
            "method": "GET",
            "params": {
                "end": game.source_window.end_ts,
                "limit": 10_000,
                "market": binding.condition_id,
                "offset": 0,
                "start": game.source_window.start_ts,
            },
        },
        source_cursor=source_cursor,
        payload=[raw],
    )
    receipt = _capture_receipt(
        manifest_sha256=manifest_sha256,
        object_sha256=object_sha256,
        request_sha256=request_sha256,
    )
    capture_game = GameCaptureResult(
        game_id=game.game_id,
        manifests=(receipt,),
        polymarket_market_count=1,
        polymarket_trade_count=1,
        polymarket_terminal_windows=(
            PolymarketTerminalWindow(
                condition_id=binding.condition_id,
                start_ts=game.source_window.start_ts,
                end_ts=game.source_window.end_ts,
                item_count=1,
                manifest_sha256=manifest_sha256,
            ),
        ),
        kalshi_market_count=0,
        kalshi_trade_proofs=(),
        kalshi_candlesticks=(),
    )
    return game, contract, raw, capture_game, (document,)


def _kalshi_verified_fixture(*, conflicting_duplicate: bool = False):
    from prediction_market.sports.nfl_x13_capture import (
        GameCaptureResult,
        KalshiCandlestickCapture,
        KalshiTradeProof,
    )

    game = _real_first_game()
    contract = next(
        value
        for value in game.contracts
        if value.venue == "kalshi" and len(value.raw_contract_ids) == 1
    )
    ticker = contract.raw_contract_ids[0]
    trade = {
        "count_fp": "3",
        "created_time": "2025-09-08T00:00:02.250000Z",
        "is_block_trade": False,
        "no_price_dollars": "0.42",
        "taker_outcome_side": "yes",
        "taker_side": "yes",
        "ticker": ticker,
        "trade_id": "verified-kalshi-trade",
        "yes_price_dollars": "0.58",
    }
    duplicate = dict(trade)
    if conflicting_duplicate:
        duplicate["yes_price_dollars"] = "0.59"
        duplicate["no_price_dollars"] = "0.41"
    candle = {
        "end_period_ts": 1_757_289_660,
        "open_interest": "20",
        "volume": "10",
        "yes_ask": {
            "close": "0.57",
            "high": "0.58",
            "low": "0.55",
            "open": "0.56",
        },
        "yes_bid": {
            "close": "0.55",
            "high": "0.56",
            "low": "0.53",
            "open": "0.54",
        },
    }
    first_manifest = _sha("4")
    second_manifest = _sha("5")
    candle_manifest = _sha("6")
    documents = (
        _verified_document(
            game_id=game.game_id,
            resource="kalshi_trades",
            manifest_sha256=first_manifest,
            object_sha256=_sha("7"),
            source_request={
                "method": "GET",
                "params": {"limit": 1000, "ticker": ticker},
            },
            source_cursor=f"ticker:{ticker};request_cursor:ROOT",
            payload={"cursor": "page-2", "trades": [trade]},
        ),
        _verified_document(
            game_id=game.game_id,
            resource="kalshi_trades",
            manifest_sha256=second_manifest,
            object_sha256=_sha("8"),
            source_request={
                "method": "GET",
                "params": {
                    "cursor": "page-2",
                    "limit": 1000,
                    "ticker": ticker,
                },
            },
            source_cursor=f"ticker:{ticker};request_cursor:page-2",
            payload={"cursor": "", "trades": [duplicate]},
        ),
        _verified_document(
            game_id=game.game_id,
            resource="kalshi_candlesticks",
            manifest_sha256=candle_manifest,
            object_sha256=_sha("9"),
            source_request={
                "method": "GET",
                "params": {
                    "end_ts": game.source_window.end_ts,
                    "period_interval": 1,
                    "start_ts": game.source_window.start_ts,
                },
            },
            source_cursor=(
                f"ticker:{ticker};start:{game.source_window.start_ts};"
                f"end:{game.source_window.end_ts};period:1"
            ),
            payload={"candlesticks": [candle], "ticker": ticker},
        ),
    )
    receipts = tuple(
        _capture_receipt(
            manifest_sha256=document.manifest_sha256,
            object_sha256=document.object_sha256,
            request_sha256=_sha(character),
        )
        for document, character in zip(
            documents,
            ("a", "b", "d"),
            strict=True,
        )
    )
    capture_game = GameCaptureResult(
        game_id=game.game_id,
        manifests=receipts,
        polymarket_market_count=0,
        polymarket_trade_count=0,
        polymarket_terminal_windows=(),
        kalshi_market_count=1,
        kalshi_trade_proofs=(
            KalshiTradeProof(
                ticker=ticker,
                page_count=2,
                raw_item_count=2,
                unique_item_count=1,
                exact_duplicate_count=1,
                terminal_cursor="",
            ),
        ),
        kalshi_candlesticks=(
            KalshiCandlestickCapture(
                ticker=ticker,
                candlestick_count=1,
                first_end_period_ts=candle["end_period_ts"],
                last_end_period_ts=candle["end_period_ts"],
                manifest_sha256=candle_manifest,
            ),
        ),
    )
    return game, contract, trade, capture_game, documents


def test_v2_observation_index_has_linear_build_and_constant_lookup_work() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        _build_observation_index_v2,
        _indexed_observed_match_v2,
    )

    count = 2_000
    observations = tuple(
        normalize_polymarket_trade(
            {
                "transactionHash": f"0xscale-{index}",
                "timestamp": 1_765_000_000 + index,
                "price": "0.61",
                "size": "4",
            },
            raw_market_id="0xscale-condition",
            logical_market_id="polymarket:0xscale-condition:winner",
            outcome="JAX",
        )
        for index in range(count)
    )

    class CountingObservations:
        def __init__(self, values):
            self.values = values
            self.visits = 0

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            self.visits += 1
            return self.values[index]

    counted = CountingObservations(observations)
    index = _build_observation_index_v2(counted)
    for observation in observations:
        matched = _indexed_observed_match_v2(
            index,
            venue=observation.venue,
            raw_market_id=observation.raw_market_id,
            logical_market_id=observation.logical_market_id,
            outcome=observation.outcome,
            native_id=observation.native_id,
            kind=observation.kind,
        )
        assert matched.observation_id == observation.observation_id

    assert counted.visits <= count + 1


def test_v2_observation_index_rejects_canonical_key_collision() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        X13MarketRecleanError,
        _build_observation_index_v2,
    )

    observation = _poly_trade()
    collision = replace(
        observation,
        observation_id=_sha("d"),
    )

    with pytest.raises(X13MarketRecleanError, match="collid"):
        _build_observation_index_v2((observation, collision))


def test_v2_observation_index_requires_one_level_observed_parent() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        X13MarketRecleanError,
        _build_observation_index_v2,
    )

    observed = _poly_trade()
    derived = derive_complement(observed, outcome="HOU")
    derived_from_derived = replace(
        derived,
        observation_id=_sha("e"),
        derived_from_observation_id=derived.observation_id,
        outcome="IND",
    )

    with pytest.raises(
        X13MarketRecleanError,
        match="derived complements",
    ):
        _build_observation_index_v2(
            (observed, derived, derived_from_derived)
        )


def test_v2_reclean_binds_each_polymarket_row_to_exact_raw_request() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        _reclean_x13_market_game_documents_v2,
    )
    from prediction_market.sports.nfl_x13_spec import canonical_sha256

    game, contract, raw, capture_game, documents = (
        _polymarket_verified_fixture()
    )

    result = _reclean_x13_market_game_documents_v2(
        game=game,
        capture_game=capture_game,
        documents=documents,
        capture_sha256=CAPTURE_SHA256,
    )

    assert result.status == "PASS"
    assert result.game_id == game.game_id
    assert len(result.normalized_observations) == 2
    assert {row.provenance for row in result.audit_rows} == {
        "observed",
        "derived_complement",
    }
    assert all(row.contract_id == contract.contract_id for row in result.audit_rows)
    assert all(row.raw_manifest_sha256 == _sha("1") for row in result.audit_rows)
    assert all(
        row.captured_license_status == "pending"
        for row in result.audit_rows
    )
    assert all(
        row.current_license_status == "research_only"
        for row in result.audit_rows
    )
    assert all(
        row.current_authority_snapshot_sha256 == _sha("a")
        for row in result.audit_rows
    )
    assert all(
        row.current_catalog_registry_sha256 == _sha("b")
        for row in result.audit_rows
    )
    assert all(
        row.current_data_license_registry_sha256 == _sha("c")
        for row in result.audit_rows
    )
    assert all(
        row.current_dataset_registry_sha256 == _sha("f")
        for row in result.audit_rows
    )
    assert all(
        row.current_experiment_registry_sha256 == _sha("d")
        for row in result.audit_rows
    )
    assert all(row.raw_object_sha256 == _sha("2") for row in result.audit_rows)
    assert all(row.raw_request_sha256 == _sha("3") for row in result.audit_rows)
    assert all(row.raw_row_index == 0 for row in result.audit_rows)
    assert all(
        row.raw_row_fingerprint == canonical_sha256(raw)
        for row in result.audit_rows
    )
    assert all(row.raw_page_index == 0 for row in result.audit_rows)
    assert all(
        row.raw_source_cursor == documents[0].source_cursor
        for row in result.audit_rows
    )
    assert all(
        row.terminal_proof == "unsaturated_time_window"
        for row in result.audit_rows
    )
    assert all(
        row.terminal_manifest_sha256 == _sha("1")
        for row in result.audit_rows
    )
    assert all(row.token_id == raw["asset"] for row in result.audit_rows)
    assert all(row.ticker is None for row in result.audit_rows)
    assert all(row.raw_request_row_available for row in result.audit_rows)
    assert all(
        row.raw_token_or_ticker_binding_available for row in result.audit_rows
    )
    assert all(row.v2_cleaning_equivalent for row in result.audit_rows)
    assert all(row.included for row in result.audit_rows)
    assert all(row.dedupe_decision == "KEEP" for row in result.audit_rows)
    assert {
        row["contract_id"] for row in result.normalized_observation_records()
    } == {contract.contract_id}
    assert result.market_cleaning_audit_records() == [
        row.to_dict() for row in result.audit_rows
    ]
    inventory = result.contract_inventory_records()
    assert any(
        row["contract_id"] == contract.contract_id
        and row["game_id"] == game.game_id
        for row in inventory
    )


def test_v2_reclean_tracks_kalshi_pages_and_drops_only_exact_duplicates() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        _reclean_x13_market_game_documents_v2,
    )

    game, contract, _, capture_game, documents = _kalshi_verified_fixture()

    result = _reclean_x13_market_game_documents_v2(
        game=game,
        capture_game=capture_game,
        documents=documents,
        capture_sha256=CAPTURE_SHA256,
    )

    assert result.status == "PASS"
    assert len(result.normalized_observations) == 3
    assert len(result.audit_rows) == 5
    trade_rows = [
        row for row in result.audit_rows if row.observation_kind == "trade"
    ]
    candle_rows = [
        row
        for row in result.audit_rows
        if row.observation_kind == "kalshi_1m_candle_bbo"
    ]
    assert [row.raw_page_index for row in trade_rows] == [0, 0, 1, 1]
    assert [row.raw_cursor_in for row in trade_rows] == [
        None,
        None,
        "page-2",
        "page-2",
    ]
    assert [row.raw_cursor_out for row in trade_rows] == [
        "page-2",
        "page-2",
        "",
        "",
    ]
    assert [row.dedupe_decision for row in trade_rows] == [
        "KEEP",
        "KEEP",
        "DROP_EXACT_DUPLICATE",
        "DROP_EXACT_DUPLICATE",
    ]
    assert [row.included for row in trade_rows] == [
        True,
        True,
        False,
        False,
    ]
    assert all(row.ticker == contract.raw_contract_ids[0] for row in trade_rows)
    assert all(row.token_id is None for row in trade_rows)
    assert all(
        row.terminal_proof == "explicit_empty_cursor"
        for row in trade_rows
    )
    assert all(
        row.terminal_manifest_sha256 == documents[1].manifest_sha256
        for row in trade_rows
    )
    assert len(candle_rows) == 1
    assert candle_rows[0].raw_page_index == 0
    assert candle_rows[0].terminal_proof == (
        "single_manifest_candlestick_count_and_interval_bounds"
    )
    assert candle_rows[0].ticker == contract.raw_contract_ids[0]


def test_v2_reclean_fails_closed_for_collision_missing_binding_and_unknown() -> None:
    from prediction_market.sports.nfl_x13_market_cleaning import (
        X13MarketRecleanError,
        _reclean_x13_market_game_documents_v2,
    )

    game, _, _, collision_capture, collision_documents = (
        _kalshi_verified_fixture(conflicting_duplicate=True)
    )
    with pytest.raises(X13MarketRecleanError, match="collid"):
        _reclean_x13_market_game_documents_v2(
            game=game,
            capture_game=collision_capture,
            documents=collision_documents,
            capture_sha256=CAPTURE_SHA256,
        )

    game, _, _, poly_capture, poly_documents = _polymarket_verified_fixture()
    missing_token = replace(
        poly_documents[0],
        payload=[
            {
                **poly_documents[0].payload[0],
                "asset": "unbound-token",
            }
        ],
    )
    with pytest.raises(X13MarketRecleanError, match="token.*unresolved"):
        _reclean_x13_market_game_documents_v2(
            game=game,
            capture_game=poly_capture,
            documents=(missing_token,),
            capture_sha256=CAPTURE_SHA256,
        )

    unknown = replace(
        poly_documents[0],
        resource="unknown_market_resource",
    )
    with pytest.raises(X13MarketRecleanError, match="unknown.*resource"):
        _reclean_x13_market_game_documents_v2(
            game=game,
            capture_game=poly_capture,
            documents=(unknown,),
            capture_sha256=CAPTURE_SHA256,
        )

    pending_current = replace(
        poly_documents[0],
        current_license_status="pending",
    )
    with pytest.raises(X13MarketRecleanError, match="current.*license"):
        _reclean_x13_market_game_documents_v2(
            game=game,
            capture_game=poly_capture,
            documents=(pending_current,),
            capture_sha256=CAPTURE_SHA256,
        )

    missing_current_hash = replace(
        poly_documents[0],
        current_dataset_registry_sha256="",
    )
    with pytest.raises(X13MarketRecleanError, match="registry"):
        _reclean_x13_market_game_documents_v2(
            game=game,
            capture_game=poly_capture,
            documents=(missing_current_hash,),
            capture_sha256=CAPTURE_SHA256,
        )


def test_v2_formal_verify_returns_immutable_payload_free_capture_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prediction_market.sports.nfl_x13_market_cleaning as cleaning

    from dataclasses import fields
    from prediction_market.sports.nfl_x13_market_cleaning import (
        VerifiedX13MarketCaptureIndexV2,
        VerifiedX13MarketManifestV2,
        verify_x13_market_capture_v2,
    )
    from prediction_market.sports.nfl_x13_pipeline import (
        load_x13_universe_artifact,
    )

    universe = load_x13_universe_artifact(UNIVERSE_ARTIFACT)
    game, _, _, capture_game, _ = _polymarket_verified_fixture()
    capture = SimpleNamespace(
        games=(capture_game,),
        manifests=capture_game.manifests,
        pointer=SimpleNamespace(
            capture_sha256=CAPTURE_SHA256,
            raw_manifest_sha256s=(_sha("1"),),
        ),
        receipt=SimpleNamespace(receipt_sha256=_sha("e")),
    )
    manifest = VerifiedX13MarketManifestV2(
        game_id=game.game_id,
        receipt=capture_game.manifests[0],
        resource="polymarket_trades",
        dataset_id="DS-POLYMARKET-PUBLIC",
        source_version="public-api-20260723",
        license_ref="O-001",
        captured_license_status="pending",
        current_license_status="research_only",
    )
    components = MappingProxyType(
        {
            "charter/catalog_registry.csv": _sha("a"),
            "registries/data_license_register.csv": _sha("b"),
            "registries/dataset_registry.csv": _sha("c"),
            "registries/experiment_registry.csv": _sha("d"),
        }
    )
    expected = VerifiedX13MarketCaptureIndexV2(
        status="PASS",
        capture_sha256=CAPTURE_SHA256,
        capture_receipt_sha256=_sha("e"),
        raw_manifest_count=1,
        raw_byte_length=1,
        manifest_hashes_verified=True,
        current_authority_snapshot_sha256=_sha("f"),
        current_authority_component_sha256s=components,
        universe=universe,
        capture=capture,
        manifests_by_game=MappingProxyType(
            {game.game_id: (manifest,)}
        ),
        batch_manifests=(),
        raw_store_root=Path("/main-repo/var/raw"),
        program_root=Path("/main-repo"),
    )
    calls: list[str] = []

    monkeypatch.setattr(
        cleaning,
        "load_x13_universe_artifact",
        lambda _: calls.append("load_universe") or universe,
    )
    monkeypatch.setattr(
        cleaning,
        "load_x13_capture",
        lambda *args, **kwargs: calls.append("load_capture") or capture,
    )
    monkeypatch.setattr(
        cleaning,
        "_build_verified_x13_market_capture_index_v2",
        lambda *args, **kwargs: calls.append("build_index") or expected,
    )

    result = verify_x13_market_capture_v2(
        universe_artifact=UNIVERSE_ARTIFACT,
        capture_pointer=Path("/main-repo/var/raw/capture-pointers/capture.json"),
        raw_store_root=Path("/main-repo/var/raw"),
        program_root=Path("/main-repo"),
    )

    assert calls == [
        "load_universe",
        "load_capture",
        "build_index",
    ]
    assert result is expected
    assert "payload" not in {field.name for field in fields(type(result))}
    assert "payload" not in {field.name for field in fields(type(manifest))}
    assert not hasattr(manifest, "payload")


def test_v2_per_game_reclean_reopens_only_selected_index_manifests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prediction_market.sports.nfl_x13_market_cleaning as cleaning

    from prediction_market.sports.nfl_x13_market_cleaning import (
        VerifiedX13MarketCaptureIndexV2,
        VerifiedX13MarketManifestV2,
        reclean_x13_verified_market_game_v2,
    )
    from prediction_market.sports.nfl_x13_pipeline import (
        load_x13_universe_artifact,
    )

    universe = load_x13_universe_artifact(UNIVERSE_ARTIFACT)
    game, _, _, capture_game, documents = _polymarket_verified_fixture()
    manifest = VerifiedX13MarketManifestV2(
        game_id=game.game_id,
        receipt=capture_game.manifests[0],
        resource="polymarket_trades",
        dataset_id="DS-POLYMARKET-PUBLIC",
        source_version="public-api-20260723",
        license_ref="O-001",
        captured_license_status="pending",
        current_license_status="research_only",
    )
    capture = SimpleNamespace(games=(capture_game,))
    index = VerifiedX13MarketCaptureIndexV2(
        status="PASS",
        capture_sha256=CAPTURE_SHA256,
        capture_receipt_sha256=_sha("e"),
        raw_manifest_count=1,
        raw_byte_length=1,
        manifest_hashes_verified=True,
        current_authority_snapshot_sha256=_sha("f"),
        current_authority_component_sha256s=MappingProxyType(
            {
                "charter/catalog_registry.csv": _sha("a"),
                "registries/data_license_register.csv": _sha("b"),
                "registries/dataset_registry.csv": _sha("c"),
                "registries/experiment_registry.csv": _sha("d"),
            }
        ),
        universe=universe,
        capture=capture,
        manifests_by_game=MappingProxyType(
            {game.game_id: (manifest,)}
        ),
        batch_manifests=(),
        raw_store_root=Path("/main-repo/var/raw"),
        program_root=Path("/main-repo"),
    )
    expected = SimpleNamespace(status="PASS", game_id=game.game_id)
    calls: list[object] = []

    monkeypatch.setattr(
        cleaning,
        "_reopen_x13_market_game_documents_v2",
        lambda actual, selected: (
            calls.append(("reopen", actual, selected)) or documents
        ),
    )
    monkeypatch.setattr(
        cleaning,
        "_reclean_x13_market_game_documents_v2",
        lambda **kwargs: (
            calls.append(("reclean", kwargs)) or expected
        ),
    )

    result = reclean_x13_verified_market_game_v2(index, game.game_id)

    assert result is expected
    assert calls[0] == ("reopen", index, game.game_id)
    assert calls[1][0] == "reclean"
    assert calls[1][1]["game"].game_id == game.game_id
    assert calls[1][1]["capture_game"] is capture_game
    assert calls[1][1]["documents"] == documents
    assert calls[1][1]["capture_sha256"] == CAPTURE_SHA256
