from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from prediction_market.sports.nfl_x13_capture import (
    CaptureHttpResponse,
    CaptureRequest,
    ImmutableRequestManifest,
)
from prediction_market.sports.nfl_x13_market import (
    SubjectRef,
    normalize_contract,
)
from prediction_market.sports.nfl_x13_spec import (
    X13_GAME_BINDINGS,
    X13_GAME_IDS,
)
from prediction_market.sports.nfl_x13_universe import (
    KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1,
    POLYMARKET_SPORTS_MARKET_TYPE_FAMILIES_V1,
    CapturedKalshiSeriesV1,
    MetadataExchangeResult,
    KalshiSeriesDiscoveryV1,
    SourceTimeWindowV1,
    X13UniverseError,
    build_kalshi_event_dependency_registry,
    build_x13_universe,
    capture_kalshi_series_metadata,
    classify_captured_kalshi_series_for_game,
    classify_cross_game_mve_market,
    classify_polymarket_event,
    derive_kalshi_event_ticker,
    discover_kalshi_series_for_game,
    exchange_and_preserve_metadata,
    publish_x13_universe,
    validate_kalshi_series_registry,
)


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _poly_market(
    market_id: int,
    *,
    market_type: str,
    condition_index: int,
    question: str = "Deliberately misleading title",
) -> dict[str, object]:
    return {
        "id": str(market_id),
        "question": question,
        "conditionId": "0x" + f"{condition_index:064x}",
        "sportsMarketType": market_type,
        "clobTokenIds": json.dumps(
            [str(condition_index * 2), str(condition_index * 2 + 1)]
        ),
        "outcomes": json.dumps(["Yes", "No"]),
        "description": f"rule-{market_id}",
        "resolutionSource": "https://www.nfl.com/",
        "gameStartTime": "2025-09-08T00:20:00Z",
        "line": 2.5,
    }


def _poly_event(
    *,
    event_id: str | None = None,
    markets: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    binding = X13_GAME_BINDINGS[0]
    return {
        "id": event_id or binding.polymarket_event_id,
        "slug": "nfl-bal-buf-2025-09-07",
        "title": "Ravens vs. Bills",
        "ticker": "nfl-bal-buf-2025-09-07",
        "sport": {"sport": "nfl"},
        "markets": markets
        or [
            {
                **_poly_market(
                    int(binding.polymarket_market_id),
                    market_type="moneyline",
                    condition_index=1,
                ),
                "conditionId": binding.polymarket_condition_id,
                "clobTokenIds": json.dumps(["101", "102"]),
                "outcomes": json.dumps(["Ravens", "Bills"]),
                "line": None,
            },
            _poly_market(
                585372,
                market_type="spreads",
                condition_index=2,
            ),
            _poly_market(
                585373,
                market_type="first_half_totals",
                condition_index=3,
            ),
            {
                **_poly_market(
                    585374,
                    market_type="parlays",
                    condition_index=4,
                ),
                "line": None,
            },
        ],
    }


def _manifest(
    index: int,
    *,
    request_sha256: str | None = None,
    body: bytes | None = None,
) -> ImmutableRequestManifest:
    object_body = body if body is not None else f"object-{index}".encode()
    return ImmutableRequestManifest(
        request_sha256=request_sha256 or _digest(f"request-{index}"),
        object_sha256=(
            "sha256:" + hashlib.sha256(object_body).hexdigest()
        ),
        manifest_sha256=_digest(f"manifest-{index}"),
        byte_length=len(object_body),
        object_path=Path(f"/raw/object-{index}.json"),
        manifest_path=Path(f"/raw/object-{index}.manifest.json"),
        response_received_at_utc=datetime(
            2026,
            7,
            25,
            0,
            0,
            index % 60,
            tzinfo=UTC,
        ),
    )


def _exchange_result(
    payload: object,
    index: int,
    *,
    request: CaptureRequest | None = None,
) -> MetadataExchangeResult:
    body = json.dumps(payload, sort_keys=True).encode()
    return MetadataExchangeResult(
        payload=payload,
        response=CaptureHttpResponse(body=body),
        manifest=_manifest(
            index,
            request_sha256=(
                request.request_sha256 if request is not None else None
            ),
            body=body,
        ),
    )


def test_metadata_exchange_seals_exact_response_at_receive_clock(
    tmp_path: Path,
) -> None:
    request = CaptureRequest(
        game_id=X13_GAME_IDS[0],
        source="polymarket",
        resource="gamma_event",
        scope_id="40828",
        url="https://gamma-api.polymarket.com/events/40828",
        params=(),
        source_cursor="event_id:40828",
        coverage="exact frozen event",
    )
    received_at = datetime(2026, 7, 25, 12, 0, 0, tzinfo=UTC)
    calls: list[str] = []

    def requester(value: CaptureRequest) -> CaptureHttpResponse:
        assert value == request
        calls.append("request")
        return CaptureHttpResponse(body=b'{"id":"40828"}')

    def clock() -> datetime:
        calls.append("clock")
        return received_at

    def writer(
        value: CaptureRequest,
        response: CaptureHttpResponse,
        raw_store_root: str | Path,
        program_root: str | Path,
        response_received_at_utc: datetime,
    ) -> ImmutableRequestManifest:
        calls.append("writer")
        assert response.body == b'{"id":"40828"}'
        assert response_received_at_utc == received_at
        return ImmutableRequestManifest(
            request_sha256=value.request_sha256,
            object_sha256=(
                "sha256:" + hashlib.sha256(response.body).hexdigest()
            ),
            manifest_sha256=_digest("sealed"),
            byte_length=len(response.body),
            object_path=Path(raw_store_root) / "object.json",
            manifest_path=Path(raw_store_root) / "manifest.json",
            response_received_at_utc=response_received_at_utc,
        )

    result = exchange_and_preserve_metadata(
        request,
        requester=requester,
        raw_store_root=tmp_path / "raw",
        program_root=tmp_path,
        response_clock=clock,
        manifest_writer=writer,
    )

    assert calls == ["request", "clock", "writer"]
    assert result.payload == {"id": "40828"}
    assert result.manifest.request_sha256 == request.request_sha256
    assert result.manifest.response_received_at_utc == received_at

    with pytest.raises(X13UniverseError, match="UTC"):
        exchange_and_preserve_metadata(
            request,
            requester=requester,
            raw_store_root=tmp_path / "raw",
            program_root=tmp_path,
            response_clock=lambda: datetime(2026, 7, 25, 12, 0, 0),
            manifest_writer=writer,
        )


def test_versioned_series_catalog_is_explicit_and_referentially_sound() -> None:
    catalog = KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1

    assert catalog.version == "v1"
    assert tuple(spec.series_ticker for spec in catalog.series) == tuple(
        sorted(spec.series_ticker for spec in catalog.series)
    )
    assert len({spec.series_ticker for spec in catalog.series}) == len(
        catalog.series
    )
    assert {
        "KXNFLGAME",
        "KXNFLSPREAD",
        "KXNFLTOTAL",
        "KXNFLTEAMTOTAL",
        "KXNFLPASSYDS",
        "KXNFLRSHYDS",
        "KXNFLRECYDS",
        "KXNFLREC",
        "KXNFLPASSTDS",
        "KXNFLANYTD",
        "KXNFLFIRSTTD",
        "KXNFL2TD",
        "KXNFLPREPACKSGP",
    } <= {spec.series_ticker for spec in catalog.series}
    assert {
        spec.family for spec in catalog.series if spec.analysis_eligible
    } <= {
        "moneyline",
        "spread",
        "total",
        "period",
        "team_total",
        "player_passing",
        "player_rushing",
        "player_receiving",
        "player_receptions",
        "player_td",
        "first_td",
        "any_td",
        "same_game_composite",
    }
    combo = next(
        spec for spec in catalog.series if spec.series_ticker == "KXNFLCOMBO"
    )
    assert combo.classification == "excluded"
    assert not combo.analysis_eligible
    assert "structured leg" in (combo.exclusion_reason or "")
    assert all(
        spec.expected_scope
        for spec in catalog.series
        if spec.classification != "composite"
    )


def test_all_series_game_event_bindings_cover_exact_20_and_jax_alias() -> None:
    catalog = KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1
    all_bindings = {
        (
            binding.native_game_id,
            series.series_ticker,
            derive_kalshi_event_ticker(series.series_ticker, binding),
        )
        for binding in X13_GAME_BINDINGS
        for series in catalog.series
        if series.classification != "composite"
    }

    assert {item[0] for item in all_bindings} == set(X13_GAME_IDS)
    assert len(all_bindings) == 20 * len(
        [
            series
            for series in catalog.series
            if series.classification != "composite"
        ]
    )
    assert (
        "2025_10_JAX_HOU",
        "KXNFLSPREAD",
        "KXNFLSPREAD-25NOV09JACHOU",
    ) in all_bindings
    assert not any("JAXHOU" in event for _, _, event in all_bindings)


def test_polymarket_classification_uses_structured_type_not_title() -> None:
    binding = X13_GAME_BINDINGS[0]
    inventory = classify_polymarket_event(
        binding,
        _poly_event(),
        manifest_sha256=_digest("gamma"),
    )

    assert [target.family for target in inventory.capture_targets] == [
        "moneyline",
        "spread",
        "period",
        "same_game_composite",
    ]
    assert [target.kind for target in inventory.capture_targets] == [
        "primitive",
        "primitive",
        "primitive",
        "same_game_composite",
    ]
    assert all(
        target.dependency_game_ids == (binding.native_game_id,)
        for target in inventory.capture_targets
    )
    assert inventory.token_bindings[0].outcomes == ("Ravens", "Bills")
    assert inventory.token_bindings[0].token_ids == ("101", "102")
    assert inventory.contracts[0].logical_market_id == (
        f"polymarket:{binding.native_game_id}:winner"
    )
    assert inventory.contracts[0].condition_id == (
        binding.polymarket_condition_id
    )
    assert inventory.contracts[0].raw_contract_ids == (
        f"polymarket:{binding.polymarket_condition_id}",
    )
    assert inventory.contracts[0].outcomes == ("Ravens", "Bills")
    assert inventory.contracts[0].comparator == "eq"
    assert inventory.contracts[1].family == "spread"
    assert inventory.contracts[1].comparator == "gt"
    assert inventory.contracts[1].line is not None
    assert inventory.contracts[2].period == "first_half"
    assert inventory.contracts[2].comparator == "gt"
    assert inventory.contracts[3].kind == "same_game_composite"
    assert (
        POLYMARKET_SPORTS_MARKET_TYPE_FAMILIES_V1["spreads"]
        == "spread"
    )

    without_sport = _poly_event()
    without_sport.pop("sport")
    without_sport["seriesSlug"] = "nfl-2025"
    assert (
        classify_polymarket_event(
            binding,
            without_sport,
            manifest_sha256=_digest("gamma-no-sport"),
        ).game_id
        == binding.native_game_id
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update({"id": "999999"}),
            "Gamma event ID",
        ),
        (
            lambda payload: payload["markets"][0].update(
                {"conditionId": "0x" + "f" * 64}
            ),
            "primary moneyline",
        ),
        (
            lambda payload: payload["markets"][1].update(
                {"sportsMarketType": "mystery_prop"}
            ),
            "unknown Polymarket sportsMarketType",
        ),
        (
            lambda payload: payload["markets"][1].update(
                {"clobTokenIds": json.dumps(["1"])}
            ),
            "token/outcome cardinality",
        ),
    ],
)
def test_polymarket_identity_and_unknown_types_fail_closed(
    mutation: object,
    message: str,
) -> None:
    payload = _poly_event()
    mutation(payload)

    with pytest.raises(X13UniverseError, match=message):
        classify_polymarket_event(
            X13_GAME_BINDINGS[0],
            payload,
            manifest_sha256=_digest("gamma"),
        )


def _series_registry_payload(
    *,
    add_unknown_game_scope: bool = False,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for spec in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series:
        row: dict[str, object] = {
            "ticker": spec.series_ticker,
            "category": "Sports",
            "settlement_sources": [
                {"name": "NFL", "url": "https://www.nfl.com/"}
            ],
        }
        if spec.expected_scope is not None:
            row["product_metadata"] = {"scope": spec.expected_scope}
        rows.append(row)
    if add_unknown_game_scope:
        rows.append(
            {
                "ticker": "KXNFLUNKNOWNPROP",
                "category": "Sports",
                "product_metadata": {"scope": "Game Props"},
                "settlement_sources": [
                    {"name": "NFL", "url": "https://www.nfl.com/"}
                ],
            }
        )
    return {"series": rows}


def test_series_registry_proves_every_catalog_identity_and_rejects_unknown() -> None:
    validated = validate_kalshi_series_registry(
        _series_registry_payload(),
        manifest_sha256=_digest("series-registry"),
    )

    assert len(validated.series_tickers) == len(
        KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
    )
    assert validated.terminal_proof == "single_complete_series_response"

    with pytest.raises(X13UniverseError, match="unknown NFL game-level series"):
        validate_kalshi_series_registry(
            _series_registry_payload(add_unknown_game_scope=True),
            manifest_sha256=_digest("series-registry"),
        )

    season_payload = _series_registry_payload()
    season_payload["series"].append(
        {
            "ticker": "KXNFLUNRELATEDSEASON",
            "category": "Sports",
            "product_metadata": {"scope": "Season Specials"},
            "settlement_sources": [
                {"name": "NFL", "url": "https://www.nfl.com/"}
            ],
        }
    )
    validate_kalshi_series_registry(
        season_payload,
        manifest_sha256=_digest("series-registry"),
    )


def _kalshi_market(
    *,
    series_ticker: str,
    event_ticker: str,
    suffix: str,
    title: str = "Misleading title is not a classifier",
) -> dict[str, object]:
    return {
        "ticker": f"{event_ticker}-{suffix}",
        "event_ticker": event_ticker,
        "market_type": "binary",
        "title": title,
        "yes_sub_title": "Yes",
        "no_sub_title": "No",
        "rules_primary": "official rule",
        "rules_secondary": "secondary rule",
        "strike_type": "greater",
        "floor_strike": 2.5,
        "custom_strike": {},
    }


def test_kalshi_series_cursor_is_terminal_and_filters_exact_game_identity() -> None:
    binding = X13_GAME_BINDINGS[0]
    spec = next(
        item
        for item in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
        if item.series_ticker == "KXNFLSPREAD"
    )
    exact_event = derive_kalshi_event_ticker(spec.series_ticker, binding)
    calls: list[CaptureRequest] = []

    def exchange(request: CaptureRequest) -> MetadataExchangeResult:
        calls.append(request)
        cursor = dict(request.params).get("cursor")
        if cursor is None:
            return _exchange_result(
                {
                    "markets": [
                        _kalshi_market(
                            series_ticker=spec.series_ticker,
                            event_ticker="KXNFLSPREAD-25SEP04DALPHI",
                            suffix="DAL3",
                        ),
                        _kalshi_market(
                            series_ticker=spec.series_ticker,
                            event_ticker=exact_event,
                            suffix="BUF2",
                        ),
                    ],
                    "cursor": "NEXT",
                },
                10,
                request=request,
            )
        assert cursor == "NEXT"
        return _exchange_result(
            {
                "markets": [
                    _kalshi_market(
                        series_ticker=spec.series_ticker,
                        event_ticker=exact_event,
                        suffix="BAL2",
                    )
                ],
                "cursor": "",
            },
            11,
            request=request,
        )

    discovery = discover_kalshi_series_for_game(
        binding,
        spec,
        exchange=exchange,
        max_pages=3,
    )

    assert [target.venue_market_id for target in discovery.capture_targets] == [
        f"{exact_event}-BAL2",
        f"{exact_event}-BUF2",
    ]
    assert all(target.family == "spread" for target in discovery.capture_targets)
    assert [contract.outcomes for contract in discovery.contracts] == [
        ("Yes", "No"),
        ("Yes", "No"),
    ]
    assert all(contract.line is not None for contract in discovery.contracts)
    assert discovery.terminal_proof == "explicit_empty_cursor"
    assert discovery.page_count == 2
    assert discovery.manifest_sha256s == (
        _digest("manifest-10"),
        _digest("manifest-11"),
    )
    assert all(
        dict(request.params).get("series_ticker") == "KXNFLSPREAD"
        for request in calls
    )
    assert all("event_ticker" not in dict(request.params) for request in calls)


def test_one_captured_series_is_reused_for_all_games_without_refetch() -> None:
    first, second = X13_GAME_BINDINGS[:2]
    spec = next(
        item
        for item in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
        if item.series_ticker == "KXNFLSPREAD"
    )
    first_event = derive_kalshi_event_ticker(spec.series_ticker, first)
    second_event = derive_kalshi_event_ticker(spec.series_ticker, second)
    calls: list[CaptureRequest] = []

    def exchange(request: CaptureRequest) -> MetadataExchangeResult:
        calls.append(request)
        return _exchange_result(
            {
                "markets": [
                    _kalshi_market(
                        series_ticker=spec.series_ticker,
                        event_ticker=first_event,
                        suffix="F",
                    ),
                    _kalshi_market(
                        series_ticker=spec.series_ticker,
                        event_ticker=second_event,
                        suffix="S",
                    ),
                ],
                "cursor": "",
            },
            14,
            request=request,
        )

    captured = capture_kalshi_series_metadata(
        spec,
        exchange=exchange,
        max_pages=2,
    )
    first_result = classify_captured_kalshi_series_for_game(first, captured)
    second_result = classify_captured_kalshi_series_for_game(second, captured)

    assert len(calls) == 1
    assert first_result.capture_targets[0].kalshi_event_ticker == first_event
    assert second_result.capture_targets[0].kalshi_event_ticker == second_event
    assert first_result.manifest_sha256s == second_result.manifest_sha256s


def test_kalshi_binary_outcomes_do_not_infer_no_from_duplicate_subtitle() -> None:
    binding = X13_GAME_BINDINGS[0]
    spec = next(
        item
        for item in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
        if item.series_ticker == "KXNFLTOTAL"
    )
    event = derive_kalshi_event_ticker(spec.series_ticker, binding)
    market = {
        **_kalshi_market(
            series_ticker=spec.series_ticker,
            event_ticker=event,
            suffix="50",
        ),
        "yes_sub_title": "Over 50.5 points",
        "no_sub_title": "Over 50.5 points",
    }

    result = discover_kalshi_series_for_game(
        binding,
        spec,
        exchange=lambda request: _exchange_result(
            {"markets": [market], "cursor": ""},
            13,
            request=request,
        ),
    )

    assert result.contracts[0].outcomes == ("Yes", "No")
    assert result.contracts[0].raw_outcome_labels == (
        (market["ticker"], "Over 50.5 points", "Over 50.5 points"),
    )
    assert result.contracts[0].comparator == "gt"


def test_universe_contract_comparators_are_normalizer_canonical() -> None:
    binding = X13_GAME_BINDINGS[0]
    poly = classify_polymarket_event(
        binding,
        _poly_event(),
        manifest_sha256=_digest("canonical-comparator-poly"),
    )
    spread_spec = next(
        item
        for item in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
        if item.series_ticker == "KXNFLSPREAD"
    )
    event = derive_kalshi_event_ticker(spread_spec.series_ticker, binding)
    kalshi = discover_kalshi_series_for_game(
        binding,
        spread_spec,
        exchange=lambda request: _exchange_result(
            {
                "markets": [
                    _kalshi_market(
                        series_ticker=spread_spec.series_ticker,
                        event_ticker=event,
                        suffix="BUF2",
                    )
                ],
                "cursor": "",
            },
            15,
            request=request,
        ),
    )

    for contract in (*poly.contracts, *kalshi.contracts):
        if contract.kind == "same_game_composite":
            continue
        normalized = normalize_contract(
            venue=contract.venue,
            venue_market_id=contract.venue_market_id,
            family=contract.family,
            period=contract.period,
            subject=SubjectRef(
                kind="game",
                canonical_id=binding.native_game_id,
                name=contract.subject,
            ),
            measure=contract.measure,
            comparator=contract.comparator,
            line=contract.line,
            outcomes=contract.outcomes,
            rule_hash=contract.rule_sha256,
            dependency_game_ids=contract.dependency_game_ids,
            kind=contract.kind,
        )
        assert normalized.comparator == contract.comparator


def test_unknown_native_kalshi_strike_type_fails_closed() -> None:
    binding = X13_GAME_BINDINGS[0]
    spec = next(
        item
        for item in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
        if item.series_ticker == "KXNFLSPREAD"
    )
    event = derive_kalshi_event_ticker(spec.series_ticker, binding)
    market = {
        **_kalshi_market(
            series_ticker=spec.series_ticker,
            event_ticker=event,
            suffix="BUF2",
        ),
        "strike_type": "mystery_native_operator",
    }

    with pytest.raises(X13UniverseError, match="strike_type"):
        discover_kalshi_series_for_game(
            binding,
            spec,
            exchange=lambda request: _exchange_result(
                {"markets": [market], "cursor": ""},
                16,
                request=request,
            ),
        )


def test_kalshi_excluded_series_is_inventory_only() -> None:
    binding = X13_GAME_BINDINGS[0]
    spec = next(
        item
        for item in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
        if item.series_ticker == "KXNFLSAFETY"
    )
    exact_event = derive_kalshi_event_ticker(spec.series_ticker, binding)

    result = discover_kalshi_series_for_game(
        binding,
        spec,
        exchange=lambda request: _exchange_result(
            {
                "markets": [
                    _kalshi_market(
                        series_ticker=spec.series_ticker,
                        event_ticker=exact_event,
                        suffix="YES",
                    )
                ],
                "cursor": "",
            },
            12,
            request=request,
        ),
    )

    assert result.capture_targets == ()
    assert len(result.excluded_contracts) == 1
    assert result.excluded_contracts[0].series_ticker == "KXNFLSAFETY"
    assert "outside approved" in result.excluded_contracts[0].reason


def test_kalshi_winner_two_tickers_share_one_logical_market() -> None:
    binding = X13_GAME_BINDINGS[0]
    spec = next(
        item
        for item in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
        if item.series_ticker == "KXNFLGAME"
    )

    def winner(ticker: str, yes_label: str, no_label: str) -> dict[str, object]:
        return {
            **_kalshi_market(
                series_ticker=spec.series_ticker,
                event_ticker=binding.kalshi_event_ticker,
                suffix=ticker.rsplit("-", 1)[1],
            ),
            "ticker": ticker,
            "yes_sub_title": yes_label,
            "no_sub_title": no_label,
        }

    result = discover_kalshi_series_for_game(
        binding,
        spec,
        exchange=lambda request: _exchange_result(
            {
                "markets": [
                    winner(binding.kalshi_away_ticker, "Baltimore", "Not Baltimore"),
                    winner(binding.kalshi_home_ticker, "Buffalo", "Not Buffalo"),
                ],
                "cursor": "",
            },
            30,
            request=request,
        ),
    )

    assert len(result.contracts) == 1
    assert result.contracts[0].logical_market_id == (
        f"kalshi:{binding.native_game_id}:winner"
    )
    assert result.contracts[0].contract_id == (
        f"kalshi:{binding.kalshi_event_ticker}"
    )
    assert result.contracts[0].venue_market_id == binding.kalshi_event_ticker
    assert result.contracts[0].raw_contract_ids == (
        binding.kalshi_away_ticker,
        binding.kalshi_home_ticker,
    )
    assert result.contracts[0].outcomes == ("BAL", "BUF")


@pytest.mark.parametrize(
    ("pages", "message"),
    [
        (
            [
                {"markets": [], "cursor": "LOOP"},
                {"markets": [], "cursor": "LOOP"},
            ],
            "repeated cursor",
        ),
        (
            [
                {
                    "markets": [
                        _kalshi_market(
                            series_ticker="KXNFLSPREAD",
                            event_ticker="OTHER-25SEP07BALBUF",
                            suffix="X",
                        )
                    ],
                    "cursor": "",
                }
            ],
            "escaped requested series",
        ),
    ],
)
def test_kalshi_cursor_and_series_identity_fail_closed(
    pages: list[dict[str, object]],
    message: str,
) -> None:
    binding = X13_GAME_BINDINGS[0]
    spec = next(
        item
        for item in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
        if item.series_ticker == "KXNFLSPREAD"
    )
    index = 0

    def exchange(request: CaptureRequest) -> MetadataExchangeResult:
        nonlocal index
        payload = pages[min(index, len(pages) - 1)]
        index += 1
        return _exchange_result(payload, 20 + index, request=request)

    with pytest.raises(X13UniverseError, match=message):
        discover_kalshi_series_for_game(
            binding,
            spec,
            exchange=exchange,
            max_pages=2,
        )


def test_structured_composite_legs_distinguish_same_and_cross_game() -> None:
    first = X13_GAME_BINDINGS[0]
    second = X13_GAME_BINDINGS[1]
    game_events = {
        first.kalshi_event_ticker: first.native_game_id,
        second.kalshi_event_ticker: second.native_game_id,
    }
    same_market = {
        "ticker": "KXNFLPREPACKSGP-X-SAME",
        "event_ticker": "KXNFLPREPACKSGP-X",
        "market_type": "binary",
        "title": "Same game",
        "custom_strike": {
            "Associated Events": (
                f"{first.kalshi_event_ticker}, "
                + derive_kalshi_event_ticker("KXNFLTOTAL", first)
            ),
            "Associated Markets": (
                f"{first.kalshi_home_ticker}, "
                + derive_kalshi_event_ticker("KXNFLTOTAL", first)
                + "-50"
            ),
            "Associated Market Sides": "Yes, No",
        },
    }
    game_events[derive_kalshi_event_ticker("KXNFLTOTAL", first)] = (
        first.native_game_id
    )
    same = classify_cross_game_mve_market(
        same_market,
        series_ticker="KXNFLPREPACKSGP",
        event_to_game_id=game_events,
    )
    assert same.kind == "same_game_composite"
    assert same.dependency_game_ids == (first.native_game_id,)

    cross_market = {
        **same_market,
        "ticker": "KXNFLPREPACK3ML-X-CROSS",
        "event_ticker": "KXNFLPREPACK3ML-X",
        "custom_strike": {
            "Associated Events": (
                f"{first.kalshi_event_ticker},"
                f"{second.kalshi_event_ticker}"
            ),
            "Associated Markets": (
                f"{first.kalshi_home_ticker},"
                f"{second.kalshi_home_ticker}"
            ),
            "Associated Market Sides": "Yes,Yes",
        },
    }
    cross = classify_cross_game_mve_market(
        cross_market,
        series_ticker="KXNFLPREPACK3ML",
        event_to_game_id=game_events,
    )
    assert cross.kind == "cross_game_inventory"
    assert cross.dependency_game_ids == tuple(
        sorted((first.native_game_id, second.native_game_id))
    )
    assert not cross.analysis_eligible

    superset_events = {
        **same_market,
        "custom_strike": {
            "Associated Events": (
                f"{first.kalshi_event_ticker},"
                f"{second.kalshi_event_ticker}"
            ),
            "Associated Markets": first.kalshi_home_ticker,
            "Associated Market Sides": "Yes",
        },
    }
    superset = classify_cross_game_mve_market(
        superset_events,
        series_ticker="KXNFLPREPACKSGP",
        event_to_game_id=game_events,
    )
    assert superset.dependency_game_ids == (first.native_game_id,)
    assert len(superset.legs) == 1

    broken = {
        **cross_market,
        "custom_strike": {
            **cross_market["custom_strike"],
            "Associated Events": (
                f"{first.kalshi_event_ticker},KXNFLGAME-UNKNOWN"
            ),
        },
    }
    with pytest.raises(X13UniverseError, match="unknown MVE leg"):
        classify_cross_game_mve_market(
            broken,
            series_ticker="KXNFLPREPACK3ML",
            event_to_game_id=game_events,
        )


def test_external_composite_dependency_requires_captured_game_event_proof() -> None:
    binding = X13_GAME_BINDINGS[0]
    game_spec = next(
        item
        for item in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
        if item.series_ticker == "KXNFLGAME"
    )
    total_spec = next(
        item
        for item in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
        if item.series_ticker == "KXNFLTOTAL"
    )
    external_suffix = "25SEP14NYGDAL"
    external_game_event = f"KXNFLGAME-{external_suffix}"
    external_total_event = f"KXNFLTOTAL-{external_suffix}"
    game_rows = (
        _kalshi_market(
            series_ticker="KXNFLGAME",
            event_ticker=binding.kalshi_event_ticker,
            suffix=binding.away_team,
        ),
        _kalshi_market(
            series_ticker="KXNFLGAME",
            event_ticker=binding.kalshi_event_ticker,
            suffix=binding.home_team,
        ),
        _kalshi_market(
            series_ticker="KXNFLGAME",
            event_ticker=external_game_event,
            suffix="NYG",
        ),
        _kalshi_market(
            series_ticker="KXNFLGAME",
            event_ticker=external_game_event,
            suffix="DAL",
        ),
    )
    total_rows = (
        _kalshi_market(
            series_ticker="KXNFLTOTAL",
            event_ticker=external_total_event,
            suffix="44",
        ),
    )
    captures = (
        CapturedKalshiSeriesV1(
            series_spec=game_spec,
            markets=game_rows,
            request_sha256s=(_digest("game-request"),),
            manifest_sha256s=(_digest("game-manifest"),),
            page_count=1,
        ),
        CapturedKalshiSeriesV1(
            series_spec=total_spec,
            markets=total_rows,
            request_sha256s=(_digest("total-request"),),
            manifest_sha256s=(_digest("total-manifest"),),
            page_count=1,
        ),
    )

    registry = build_kalshi_event_dependency_registry(captures)

    assert registry[binding.kalshi_event_ticker] == binding.native_game_id
    assert registry[external_game_event] == (
        f"kalshi_game:{external_game_event}"
    )
    assert registry[external_total_event] == (
        f"kalshi_game:{external_game_event}"
    )

    unproved = f"KXNFLTOTAL-25SEP21UNKNOWN"
    assert unproved not in registry

    incomplete_game_capture = CapturedKalshiSeriesV1(
        series_spec=game_spec,
        markets=game_rows[:-1],
        request_sha256s=(_digest("bad-game-request"),),
        manifest_sha256s=(_digest("bad-game-manifest"),),
        page_count=1,
    )
    with pytest.raises(X13UniverseError, match="two outcome tickers"):
        build_kalshi_event_dependency_registry(
            (incomplete_game_capture, captures[1])
        )


def test_unrelated_malformed_composite_does_not_poison_frozen_game() -> None:
    binding = X13_GAME_BINDINGS[0]
    unrelated = X13_GAME_BINDINGS[1]
    spec = next(
        item
        for item in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series
        if item.series_ticker == "KXNFLPREPACK"
    )
    market = {
        "ticker": "KXNFLPREPACK-X-UNRELATED",
        "event_ticker": "KXNFLPREPACK-X",
        "market_type": "binary",
        "title": "Unrelated",
        "custom_strike": {
            "Associated Events": unrelated.kalshi_event_ticker,
            "Associated Markets": (
                f"{unrelated.kalshi_home_ticker},"
                f"{unrelated.kalshi_away_ticker}"
            ),
            "Associated Market Sides": "Yes",
        },
    }

    result = discover_kalshi_series_for_game(
        binding,
        spec,
        exchange=lambda request: _exchange_result(
            {"markets": [market], "cursor": ""},
            31,
            request=request,
        ),
    )

    assert result.capture_targets == ()
    assert result.cross_game_inventory == ()


def _minimal_polymarket_inventory(binding_index: int) -> object:
    binding = X13_GAME_BINDINGS[binding_index]
    payload = {
        "id": binding.polymarket_event_id,
        "slug": (
            f"nfl-{binding.away_team.lower()}-"
            f"{binding.home_team.lower()}-{binding.game_date.isoformat()}"
        ),
        "title": f"{binding.away_team} vs. {binding.home_team}",
        "ticker": f"nfl-{binding_index}",
        "sport": {"sport": "nfl"},
        "markets": [
            {
                **_poly_market(
                    int(binding.polymarket_market_id),
                    market_type="moneyline",
                    condition_index=1000 + binding_index,
                ),
                "conditionId": binding.polymarket_condition_id,
                "clobTokenIds": json.dumps(
                    [
                        str(10_000 + binding_index * 2),
                        str(10_001 + binding_index * 2),
                    ]
                ),
                "outcomes": json.dumps(
                    [binding.away_team, binding.home_team]
                ),
                "line": None,
            }
        ],
    }
    return classify_polymarket_event(
        binding,
        payload,
        manifest_sha256=_digest(f"poly-{binding.native_game_id}"),
    )


def _minimal_kalshi_discoveries(binding_index: int) -> tuple[object, ...]:
    binding = X13_GAME_BINDINGS[binding_index]
    result: list[object] = []
    for spec in KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series:
        if spec.series_ticker == "KXNFLGAME":
            result.append(
                discover_kalshi_series_for_game(
                    binding,
                    spec,
                    exchange=lambda request, binding=binding: _exchange_result(
                        {
                            "markets": [
                                {
                                    **_kalshi_market(
                                        series_ticker="KXNFLGAME",
                                        event_ticker=binding.kalshi_event_ticker,
                                        suffix=binding.away_team,
                                    ),
                                    "ticker": binding.kalshi_away_ticker,
                                    "yes_sub_title": binding.away_team,
                                    "no_sub_title": f"Not {binding.away_team}",
                                },
                                {
                                    **_kalshi_market(
                                        series_ticker="KXNFLGAME",
                                        event_ticker=binding.kalshi_event_ticker,
                                        suffix=binding.home_team,
                                    ),
                                    "ticker": binding.kalshi_home_ticker,
                                    "yes_sub_title": binding.home_team,
                                    "no_sub_title": f"Not {binding.home_team}",
                                },
                            ],
                            "cursor": "",
                        },
                        100 + binding_index,
                        request=request,
                    ),
                )
            )
        else:
            result.append(
                KalshiSeriesDiscoveryV1(
                    game_id=binding.native_game_id,
                    series_ticker=spec.series_ticker,
                    capture_targets=(),
                    contracts=(),
                    excluded_contracts=(),
                    cross_game_inventory=(),
                    manifest_sha256s=(
                        _digest(
                            f"{binding.native_game_id}:{spec.series_ticker}"
                        ),
                    ),
                    page_count=1,
                )
            )
    return tuple(result)


def _complete_batch_inputs() -> tuple[object, object, object, object]:
    poly = tuple(
        _minimal_polymarket_inventory(index)
        for index in range(len(X13_GAME_BINDINGS))
    )
    kalshi = tuple(
        item
        for index in range(len(X13_GAME_BINDINGS))
        for item in _minimal_kalshi_discoveries(index)
    )
    windows = {
        game_id: SourceTimeWindowV1(
            start_ts=1_750_000_000 + index * 10_000,
            end_ts=1_750_010_000 + index * 10_000,
        )
        for index, game_id in enumerate(X13_GAME_IDS)
    }
    registry = validate_kalshi_series_registry(
        _series_registry_payload(),
        manifest_sha256=_digest("series-registry-complete"),
    )
    return poly, kalshi, windows, registry


def test_build_exact_20_game_universe_and_capture_plans(tmp_path: Path) -> None:
    poly, kalshi, windows, registry = _complete_batch_inputs()

    batch = build_x13_universe(
        polymarket_inventories=poly,
        kalshi_discoveries=kalshi,
        source_windows=windows,
        series_registry_proof=registry,
    )

    assert batch.game_ids == X13_GAME_IDS
    assert len(batch.games) == 20
    assert len(batch.capture_plans) == 20
    assert all(
        {target.venue for target in plan.targets}
        == {"polymarket", "kalshi"}
        for plan in batch.capture_plans
    )
    assert all(
        len(game.kalshi_series_tickers)
        == len(KALSHI_NFL_SINGLE_GAME_SERIES_CATALOG_V1.series)
        for game in batch.games
    )
    assert all(game.unknown_contract_ids == () for game in batch.games)
    assert batch.universe_id.startswith("sha256:")

    artifact = publish_x13_universe(
        batch,
        tmp_path / "x13-universe-plan-v1.json",
    )
    payload = json.loads(artifact.read_text())
    assert payload["game_ids"] == list(X13_GAME_IDS)
    assert payload["schema_version"] == "nfl-x13-universe-v1"
    assert len(payload["games"]) == 20
    published_universe_id = payload.pop("universe_id")
    assert published_universe_id == (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )
    with pytest.raises(X13UniverseError, match="already exists"):
        publish_x13_universe(batch, artifact)


def test_batch_builder_rejects_missing_series_before_plan_publish() -> None:
    poly, kalshi, windows, registry = _complete_batch_inputs()
    missing = tuple(
        discovery
        for discovery in kalshi
        if not (
            discovery.game_id == X13_GAME_IDS[0]
            and discovery.series_ticker == "KXNFLSPREAD"
        )
    )

    with pytest.raises(X13UniverseError, match="series coverage"):
        build_x13_universe(
            polymarket_inventories=poly,
            kalshi_discoveries=missing,
            source_windows=windows,
            series_registry_proof=registry,
        )
