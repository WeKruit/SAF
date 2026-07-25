import assert from "node:assert/strict";
import test from "node:test";

import * as dashboardViewModel from "../lib/dashboard-view-model.mjs";

const {
  buildGameIndexRows,
  filterAssociationRows,
  filterContracts,
  marketSeriesView,
  replayEvidenceAt,
  lineageEntries,
  evidenceStateClass,
} = dashboardViewModel;

test("market selection has one atomic state transition function", () => {
  assert.equal(
    typeof dashboardViewModel.reduceMarketSelection,
    "function",
    "selection id, evidence, loading, and error need one reducer",
  );
  assert.equal(
    typeof dashboardViewModel.reconcileMarketSelection,
    "function",
    "contract filters need the same atomic clear transition",
  );
});

test("clearing a market invalidates its request and clears all evidence state", () => {
  const state = {
    id: "A",
    market: { payload: { logical_market_id: "A" } },
    error: "old error",
    loading: false,
    generation: 3,
  };
  assert.deepEqual(
    dashboardViewModel.reduceMarketSelection(state, {
      type: "clear",
      generation: 4,
    }),
    {
      id: "",
      market: null,
      error: "",
      loading: false,
      generation: 4,
    },
  );
});

test("failed B never leaves A evidence behind and a later success clears the error", () => {
  const loadedA = {
    id: "A",
    market: { payload: { logical_market_id: "A" } },
    error: "",
    loading: false,
    generation: 1,
  };
  const requestingB = dashboardViewModel.reduceMarketSelection(loadedA, {
    type: "request",
    id: "B",
    generation: 2,
  });
  assert.deepEqual(requestingB, {
    id: "B",
    market: null,
    error: "",
    loading: true,
    generation: 2,
  });

  const failedB = dashboardViewModel.reduceMarketSelection(requestingB, {
    type: "failure",
    id: "B",
    generation: 2,
    error: "B unavailable",
  });
  assert.equal(failedB.id, "B");
  assert.equal(failedB.market, null);
  assert.equal(failedB.error, "B unavailable");
  assert.equal(failedB.loading, false);

  assert.equal(
    dashboardViewModel.reduceMarketSelection(failedB, {
      type: "success",
      id: "A",
      generation: 1,
      market: loadedA.market,
    }),
    failedB,
    "stale A completion must not change B state",
  );

  const loadedB = dashboardViewModel.reduceMarketSelection(failedB, {
    type: "success",
    id: "B",
    generation: 3,
    market: { payload: { logical_market_id: "B" } },
  });
  assert.equal(loadedB.id, "B");
  assert.equal(loadedB.market.payload.logical_market_id, "B");
  assert.equal(loadedB.error, "");
  assert.equal(loadedB.loading, false);
});

test("filter exclusion clears the selected market and its evidence atomically", () => {
  const selected = {
    id: "A",
    market: { payload: { logical_market_id: "A" } },
    error: "",
    loading: false,
    generation: 7,
  };
  const filteredContracts = [{ logical_market_id: "B" }];
  assert.deepEqual(
    dashboardViewModel.reconcileMarketSelection(
      selected,
      filteredContracts,
      8,
    ),
    {
      id: "",
      market: null,
      error: "",
      loading: false,
      generation: 8,
    },
  );
});

test("evidence state classification is whitelist-only and fail closed", () => {
  assert.equal(evidenceStateClass("OBSERVED"), "state-observed");
  assert.equal(evidenceStateClass("PASS"), "state-observed");
  assert.equal(evidenceStateClass("ORDER_AMBIGUOUS"), "state-ambiguous");
  for (const value of [
    "unknown_future_state",
    "NOT_AUDITED",
    "AWAITING_PRIVATE_CATALOG",
    "NO_POST_TRADE",
    undefined,
  ]) {
    assert.equal(evidenceStateClass(value), "state-unavailable");
  }
});

test("game index exposes episodes, venues, and the per-game audit gate", () => {
  const rows = buildGameIndexRows({
    payload: {
      games: [
        {
          game_id: "2025_14_DAL_DET",
          away_team: "DAL",
          home_team: "DET",
          episode_count: 187,
          venues: ["kalshi", "polymarket"],
          audit: { layer_a: "PASS_SOURCE_TIME_ONLY" },
        },
      ],
    },
  });
  assert.deepEqual(rows[0], {
    gameId: "2025_14_DAL_DET",
    awayTeam: "DAL",
    homeTeam: "DET",
    awayScore: undefined,
    homeScore: undefined,
    eventCount: undefined,
    episodeCount: 187,
    contractCount: undefined,
    venues: ["kalshi", "polymarket"],
    auditGate: "PASS_SOURCE_TIME_ONLY",
    status: undefined,
  });
});

test("replay evidence uses canonical personnel coverage and the matching cumulative ledger", () => {
  const core = {
    payload: {
      personnel: {
        coverage: "research_only",
        complete_11v11_rows: 181,
        missing_rows: 11,
        semantics: "observed on-field roster per play",
      },
      events: [
        {
          play_id: "500",
          order_sequence: 22,
          offense_personnel: "11 personnel",
          defense_personnel: "nickel",
          offense_names: ["D.Prescott", "C.Lamb"],
          defense_names: ["A.Hutchinson"],
          player_roles: { passer: "D.Prescott" },
        },
      ],
      stat_ledger: [
        {
          through_play_id: "500",
          through_order_sequence: 22,
          team_scores: { DAL: 3, DET: 3 },
          period_scores: { 1: { DAL: 3, DET: 3 } },
          player_stats: {
            "D.Prescott": {
              passing_yards: 42,
              passing_touchdowns: 1,
              rushing_touchdowns: 2,
              receiving_touchdowns: 0,
            },
          },
        },
      ],
    },
  };
  const view = replayEvidenceAt(core, 0);
  assert.equal(view.personnel.coverage, "research_only");
  assert.equal(view.personnel.completeRows, 181);
  assert.equal(view.offense.package, "11 personnel");
  assert.deepEqual(view.offense.names, ["D.Prescott", "C.Lamb"]);
  assert.deepEqual(view.ledger.teamScores, { DAL: 3, DET: 3 });
  assert.equal(view.ledger.rolePlayerStats[0].passing_yards, 42);
  assert.equal(view.ledger.rolePlayerStats[0].canonicalTouchdowns, 3);
});

test("market series exposes observed trades and source Kalshi bid/ask OHLC", () => {
  const market = {
    payload: {
      series: [
        {
          outcome: "Yes",
          series_kind: "trade_range",
          provenance: "observed",
          points: [{ vwap: "0.35" }],
        },
        {
          outcome: "No",
          series_kind: "trade_range",
          provenance: "derived_complement",
          points: [{ vwap: "0.65" }],
        },
        {
          outcome: "Yes",
          series_kind: "bbo_candle",
          provenance: "observed",
          points: [
            {
              bid_ohlc: { open: "0.34", high: "0.35", low: "0.33", close: "0.34" },
              ask_ohlc: { open: "0.36", high: "0.39", low: "0.35", close: "0.38" },
            },
          ],
        },
      ],
    },
  };
  const view = marketSeriesView(market);
  assert.equal(view.trades.length, 2);
  assert.equal(view.bbo.length, 1);
  assert.deepEqual(view.bbo[0].points[0].bid, {
    open: 0.34,
    high: 0.35,
    low: 0.33,
    close: 0.34,
  });
  assert.deepEqual(view.bbo[0].points[0].ask, {
    open: 0.36,
    high: 0.39,
    low: 0.35,
    close: 0.38,
  });
});

test("null trade VWAP stays unavailable and discontinuity is explicit", () => {
  const view = marketSeriesView({
    payload: {
      series: [
        {
          outcome: "DAL",
          series_kind: "trade_range",
          provenance: "observed",
          points: [
            {
              vwap: null,
              contains_discontinuity: true,
              source_start: "2025-12-05T01:14:42Z",
            },
            {
              vwap: "0.54",
              contains_discontinuity: false,
              source_start: "2025-12-05T01:14:43Z",
            },
          ],
        },
      ],
    },
  });
  assert.equal(view.trades[0].points[0].vwap, null);
  assert.equal(view.trades[0].points[0].available, false);
  assert.equal(view.trades[0].unavailablePointCount, 1);
  assert.equal(view.trades[0].discontinuityCount, 1);
  assert.equal(view.trades[0].points[1].vwap, 0.54);
});

test("null BBO fields remain unavailable rather than becoming zero", () => {
  const view = marketSeriesView({
    payload: {
      series: [
        {
          outcome: "DAL",
          series_kind: "bbo_candle",
          provenance: "observed",
          points: [
            {
              bid_ohlc: {
                open: null,
                high: "0.56",
                low: "0.53",
                close: "0.55",
              },
              ask_ohlc: {
                open: "0.58",
                high: "0.59",
                low: "0.56",
                close: null,
              },
              contains_discontinuity: true,
            },
          ],
        },
      ],
    },
  });
  assert.equal(view.bbo[0].points[0].bid.open, null);
  assert.equal(view.bbo[0].points[0].ask.close, null);
  assert.equal(view.bbo[0].points[0].available, false);
  assert.equal(view.bbo[0].unavailablePointCount, 1);
  assert.equal(view.bbo[0].discontinuityCount, 1);
});

test("all BBO outcome series remain distinct", () => {
  const candle = {
    bid_ohlc: { open: "0.40", high: "0.42", low: "0.39", close: "0.41" },
    ask_ohlc: { open: "0.43", high: "0.44", low: "0.42", close: "0.43" },
  };
  const unavailableDiscontinuousCandle = {
    bid_ohlc: { open: null, high: "0.42", low: "0.39", close: "0.41" },
    ask_ohlc: { open: "0.43", high: "0.44", low: "0.42", close: "0.43" },
    contains_discontinuity: true,
    source_order_ambiguous: true,
  };
  const view = marketSeriesView({
    payload: {
      series: [
        {
          outcome: "DAL",
          series_kind: "bbo_candle",
          provenance: "observed",
          points: [
            { ...candle, source_order_ambiguous: true },
            unavailableDiscontinuousCandle,
          ],
        },
        {
          outcome: "DET",
          series_kind: "bbo_candle",
          provenance: "observed",
          points: [
            { ...candle, source_order_ambiguous: true },
            unavailableDiscontinuousCandle,
          ],
        },
      ],
    },
  });
  assert.deepEqual(view.bbo.map((series) => series.outcome), ["DAL", "DET"]);
  for (const series of view.bbo) {
    assert.deepEqual(
      series.points.map((point) => point.status),
      ["ambiguous", "ambiguous"],
      "source ambiguity outranks numeric availability and discontinuity",
    );
    assert.equal(series.ambiguousPointCount, 2);
    assert.equal(series.unavailablePointCount, 1);
    assert.equal(series.discontinuityCount, 1);
    assert.equal(
      series.points[0].available,
      true,
      "numeric OHLC remains inspectable but must not be classified observed",
    );
  }
});

test("venue and market-state selectors constrain contracts", () => {
  const contracts = [
    { venue: "kalshi", family: "moneyline", analysis_eligible: true },
    { venue: "polymarket", family: "moneyline", analysis_eligible: false },
    { venue: "kalshi", family: "spread", analysis_eligible: true },
  ];
  assert.deepEqual(
    filterContracts(contracts, {
      family: "moneyline",
      venue: "kalshi",
      state: "analysis_eligible",
    }),
    [contracts[0]],
  );
});

test("controlled association status filter is applied with other evidence locks", () => {
  const rows = [
    {
      delay_scenario_seconds: 0,
      horizon_seconds: 10,
      validity_status: "OBSERVED",
      order_ambiguous: false,
      contaminated: false,
    },
    {
      delay_scenario_seconds: 0,
      horizon_seconds: 10,
      validity_status: "NO_PRE_EVENT_TRADE",
      order_ambiguous: false,
      contaminated: false,
    },
  ];
  assert.deepEqual(
    filterAssociationRows(rows, {
      delay: "0",
      horizon: "10",
      includeAmbiguous: false,
      includeContaminated: false,
      status: "OBSERVED",
    }),
    [rows[0]],
  );
});

test("lineage exposes builder, analysis lock, state ledger, and raw manifest count", () => {
  const rows = lineageEntries({
    payload: {
      lineage: {
        builder_version: "nfl-x13-pipeline-v2",
        analysis_lock: `sha256:${"1".repeat(64)}`,
        game_state_ledger_sha256: `sha256:${"2".repeat(64)}`,
        raw_manifest_hashes: [`sha256:${"3".repeat(64)}`],
      },
    },
  });
  assert.deepEqual(rows.slice(0, 4).map((row) => row.label), [
    "Builder",
    "Analysis lock",
    "Game-state ledger",
    "Raw manifests",
  ]);
  assert.equal(rows[3].value, "1 verified hashes");
  assert.equal(rows[4].label, "Raw hash 1");
  assert.equal(rows[4].value, `sha256:${"3".repeat(64)}`);
});
