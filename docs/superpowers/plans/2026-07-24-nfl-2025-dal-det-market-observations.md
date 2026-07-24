# NFL 2025 DAL@DET Market Observation Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve exact, query-bound historical Polymarket and Kalshi observations for `2025_14_DAL_DET`, then emit a venue-local mapping/time-semantics audit without calculating market reaction, alpha, executable fills, or cross-venue prices.

**Architecture:** The capture layer is split by venue. Polymarket preserves a Gamma event response plus one condition-scoped Data API trades window. Kalshi reuses the raw-first historical REST client with query-bound pagination for two winner tickers and their one-minute candles. An audit binds stable venue IDs to the frozen NFL game identity, but intentionally leaves the game-event-to-market time join for the following phase.

**Tech Stack:** Python 3.12, httpx, existing immutable static-store, pytest.

---

## File structure

- Create: `src/prediction_market/polymarket_history.py` — raw-first bounded HTTP capture for a closed Gamma event and one condition-scoped trades window.
- Modify: `src/prediction_market/kalshi_history.py` — allow a validated, immutable query mapping to persist across a bounded historical cursor chain.
- Create: `src/prediction_market/sports/nfl_market_observation.py` — validates one NFL game’s stable venue IDs and reports timestamp semantics only.
- Create: `tests/data/test_polymarket_history.py` — mock-HTTP tests for raw-first preservation, condition filtering, timestamp window bounds, and offset-cap rejection.
- Modify: `tests/data/test_kalshi_history.py` — verifies a ticker/time query remains bound to every cursor-page manifest.
- Create: `tests/sports/test_nfl_market_observation.py` — mapping/time-semantics report test with fixtures; explicitly rejects a reaction or execution claim.
- Create at execution time: `artifacts/market-observation/nfl/nfl_2025_14_dal_det_market_mapping_v0.json` — derived audit with raw manifest references only.

### Task 1: Add bounded Polymarket historical capture

**Files:**
- Create: `tests/data/test_polymarket_history.py`
- Create: `src/prediction_market/polymarket_history.py`

- [x] **Step 1: Write failing fixture tests**

```python
def test_capture_preserves_gamma_and_condition_scoped_trades_raw_first(tmp_path: Path) -> None:
    capture = capture_polymarket_history(
        event_slug="nfl-dal-det-2025-12-04",
        condition_id="0x" + "a" * 64,
        start_ts=100,
        end_ts=200,
        store_root=tmp_path,
        program_root=PROJECT_ROOT,
        fetched_at=FETCHED_AT,
        client=_client(),
    )
    assert capture.trade_count == 2
    assert capture.trades.record.manifest.source_request["params"]["market"] == "0x" + "a" * 64
    assert capture.event.record.manifest.license_status == "pending"


def test_capture_rejects_a_full_10000_row_trades_page_after_preserving_it(tmp_path: Path) -> None:
    with pytest.raises(PolymarketHistoryError, match="subdivide"):
        capture_polymarket_history(..., client=_full_page_client())
    assert len(list((tmp_path / "raw").rglob("*.json"))) == 2
```

- [x] **Step 2: Run tests to verify RED**

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/data/test_polymarket_history.py -q`

Expected: import failure because `polymarket_history` does not exist.

- [x] **Step 3: Implement minimal capture**

```python
def capture_polymarket_history(
    *, event_slug: str, condition_id: str, start_ts: int, end_ts: int,
    store_root: str | Path, program_root: str | Path, fetched_at: datetime,
    client: httpx.Client | None = None,
) -> PolymarketHistoricalCapture:
    ...
```

The implementation must use `Accept-Encoding: identity`; preserve Gamma bytes before shape validation; preserve Data API bytes before trade validation; bind exact query params in the static manifest; reject condition mismatch, timestamp outside `[start_ts, end_ts]`, duplicate transaction/outcome/asset/timestamp/side records with unequal content, and 10,000 returned trades (which would require a split window). It must mark `DS-POLYMARKET-PUBLIC` / `O-001` as `pending` and must not fetch CLOB price history or order-book data.

- [x] **Step 4: Run GREEN**

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/data/test_polymarket_history.py -q`

Expected: PASS.

### Task 2: Bind Kalshi historical queries to cursor pages

**Files:**
- Modify: `tests/data/test_kalshi_history.py`
- Modify: `src/prediction_market/kalshi_history.py`

- [x] **Step 1: Write failing query propagation test**

```python
def test_trade_backfill_binds_ticker_and_time_query_on_every_page(tmp_path: Path) -> None:
    result = backfill_kalshi_historical(
        "trades", query={"ticker": "KXNFLGAME-25DEC04DALDET-DET", "min_ts": 100, "max_ts": 200},
        store_root=tmp_path, program_root=PROJECT_ROOT, fetched_at=FETCHED_AT,
        client=_client(_trade_pages()), page_limit=2, max_pages=2,
    )
    assert [page.record.manifest.source_request["params"] for page in result.pages] == [
        {"limit": 2, "ticker": "KXNFLGAME-25DEC04DALDET-DET", "min_ts": 100, "max_ts": 200},
        {"limit": 2, "ticker": "KXNFLGAME-25DEC04DALDET-DET", "min_ts": 100, "max_ts": 200, "cursor": "next"},
    ]
```

- [x] **Step 2: Run test to verify RED**

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/data/test_kalshi_history.py::test_trade_backfill_binds_ticker_and_time_query_on_every_page -q`

Expected: FAIL because `backfill_kalshi_historical` has no `query` argument.

- [x] **Step 3: Implement exact query validation and persistence**

Accept only `ticker`, `min_ts`, `max_ts`, and `is_block_trade` for `trades`; accept only `tickers`, `event_ticker`, `series_ticker`, and `mve_filter` for `markets`. Reject unsupported keys, unsafe ticker text, booleans outside `is_block_trade`, non-integer timestamps, and `min_ts > max_ts`. Add the exact validated query to every request and coverage value; retain the existing no-query behavior unchanged.

- [x] **Step 4: Run GREEN**

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/data/test_kalshi_history.py -q`

Expected: PASS.

### Task 3: Audit the venue-local game mapping and capture real bounded objects

**Files:**
- Create: `tests/sports/test_nfl_market_observation.py`
- Create: `src/prediction_market/sports/nfl_market_observation.py`
- Create: `artifacts/market-observation/nfl/nfl_2025_14_dal_det_market_mapping_v0.json`

- [x] **Step 1: Write the failing mapping-boundary test**

```python
def test_market_mapping_declares_venue_timestamps_without_reaction_claim() -> None:
    report = build_nfl_market_mapping(...)
    assert report["canonical_game_id"] == "game_nflverse_2025_14_DAL_DET"
    assert report["venue_time_semantics"]["polymarket"] == "trade.timestamp_epoch_seconds"
    assert report["venue_time_semantics"]["kalshi"] == "trade.created_time_and_candle.end_period_ts"
    assert report["evidence_boundary"]["reaction_measured"] is False
    assert report["evidence_boundary"]["cross_venue_comparison"] is False
```

- [x] **Step 2: Run test to verify RED**

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/sports/test_nfl_market_observation.py -q`

Expected: import failure because the audit module does not exist.

- [x] **Step 3: Implement the mapping audit and capture real public objects**

The real capture parameters are fixed:

```text
NFL game: 2025_14_DAL_DET
Polymarket event slug: nfl-dal-det-2025-12-04
Polymarket condition: 0xe6f492eb15583ae324d43e987e3ad1b7bd1a86690b3b62a2843162278ff7af0e
Time window: [1764897300, 1764924000]
Kalshi markets: KXNFLGAME-25DEC04DALDET-DET and KXNFLGAME-25DEC04DALDET-DAL
Kalshi candle interval: 1 minute
```

Use the real main raw root, not a worktree symlink. Preserve all remote bytes and manifests there. The derived audit must reference source object/manifest hashes, list the two Kalshi winner tickers separately, state that Polymarket trades are public trade observations rather than L2, state that Kalshi candles are interval-ending aggregates, and set every formal evidence/reaction/alpha/execution flag to false. It must not write below `var/raw` except through the static-store capture functions.

- [x] **Step 4: Run focused verification**

Run:

```bash
PYTHONPATH=src ../../.venv/bin/python -m pytest \
  tests/data/test_polymarket_history.py \
  tests/data/test_kalshi_history.py \
  tests/sports/test_nfl_market_observation.py \
  tests/sports/test_nfl_game_replay.py -q
```

Expected: PASS. The output artifact must explicitly say `reaction_measured=false`, `pit_validated=false`, and `formal_result_eligible=false`.
