"use client";

import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { DashboardDataClient } from "@/lib/dashboard-client.mjs";
import {
  buildGameIndexRows,
  episodeIdAtEvent,
  evidenceStateClass,
  filterAssociationRows,
  filterContracts,
  lineageEntries,
  marketSeriesView,
  parseDashboardReference,
  reconcileMarketSelection,
  reduceMarketSelection,
  replayEvidenceAt,
  resolveEpisodeEventIndex,
  resolveMarketReference,
} from "@/lib/dashboard-view-model.mjs";

const frozenGames = [
  "2025_01_BAL_BUF",
  "2025_03_ARI_SF",
  "2025_03_CIN_MIN",
  "2025_03_LA_PHI",
  "2025_03_NYJ_TB",
  "2025_04_GB_DAL",
  "2025_04_PHI_TB",
  "2025_08_CLE_NE",
  "2025_08_NYJ_CIN",
  "2025_09_CHI_CIN",
  "2025_10_JAX_HOU",
  "2025_13_NO_MIA",
  "2025_14_DAL_DET",
  "2025_14_PHI_LAC",
  "2025_14_SEA_ATL",
  "2025_16_LA_SEA",
  "2025_18_KC_LV",
  "2025_19_LA_CAR",
  "2025_20_BUF_DEN",
  "2025_20_HOU_NE",
];

// Dashboard projections are runtime-verified, schema-versioned JSON documents.
// Their deliberately open payload shape is narrowed at each rendering boundary.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type JsonRecord = Record<string, any>;

function gameTeams(gameId: string) {
  const parts = gameId.split("_");
  return { away: parts.at(-2) ?? "AWAY", home: parts.at(-1) ?? "HOME" };
}

function clockLabel(event?: JsonRecord) {
  if (!event) return "Q— · —:——";
  const remaining = Number(event.clock_seconds_remaining ?? 0);
  return `Q${event.quarter ?? "—"} · ${Math.floor(remaining / 60)}:${String(
    remaining % 60,
  ).padStart(2, "0")}`;
}

function shortTime(value?: string | null) {
  if (!value) return "source time unavailable";
  return new Date(value).toISOString().slice(11, 23) + "Z";
}

function EventPill({ event }: { event: JsonRecord }) {
  const flags = [
    event.touchdown && "TD",
    event.turnover && "TURNOVER",
    event.penalty && "PENALTY",
    event.timeout && "TIMEOUT",
    event.review && "REVIEW",
    event.between_play_roster_change && "ROSTER Δ",
  ].filter(Boolean);
  return (
    <div className="event-flags">
      <span className="play-family">{event.play_family ?? "administrative"}</span>
      {flags.map((flag) => (
        <span key={String(flag)} className="event-flag">
          {flag}
        </span>
      ))}
    </div>
  );
}

function ohlcLabel(value?: JsonRecord) {
  if (!value) return "unavailable";
  const fields = ["open", "high", "low", "close"].map((field) =>
    value[field] === null || value[field] === undefined
      ? "unavailable"
      : String(value[field]),
  );
  return `O ${fields[0]} / H ${fields[1]} / L ${fields[2]} / C ${fields[3]}`;
}

function FootballField({ event }: { event?: JsonRecord }) {
  const coordinate = Math.max(
    0,
    Math.min(100, Number(event?.field_coordinate_0_100 ?? 50)),
  );
  const goingRight = event?.direction !== "left";
  return (
    <div className="field-shell" aria-label="Football field position">
      <div className="field-meta">
        <span>{event?.offense_team ?? "OFF"} offense</span>
        <strong>{event?.yardline_100 == null ? "—" : `${event.yardline_100} YTG`}</strong>
        <span>{event?.defense_team ?? "DEF"} defense</span>
      </div>
      <div className="football-field">
        <div className="endzone endzone-away">{event?.away_team ?? "AWAY"}</div>
        <div className="playing-surface">
          {Array.from({ length: 9 }, (_, index) => (
            <i key={index} style={{ left: `${10 + index * 10}%` }} />
          ))}
          <div
            className="line-of-scrimmage"
            style={{ left: `${coordinate}%` }}
          />
          <div className="ball" style={{ left: `${coordinate}%` }}>
            <span aria-hidden="true">◆</span>
          </div>
          <div
            className={`direction-arrow ${goingRight ? "right" : "left"}`}
            style={{ left: `${coordinate}%` }}
          >
            {goingRight ? "→" : "←"}
          </div>
        </div>
        <div className="endzone endzone-home">{event?.home_team ?? "HOME"}</div>
      </div>
      <p className="field-caption">
        Source field coordinate · offense moving {goingRight ? "right" : "left"}
      </p>
    </div>
  );
}

function MarketPath({ market }: { market: JsonRecord | null }) {
  const view = marketSeriesView(market);
  const outcomeSeries: JsonRecord[] = view.trades;
  const bboSeries: JsonRecord[] = view.bbo;
  return (
    <>
      <div className="market-path">
        <div className="chart-axis">
          <span>1.00</span>
          <span>.50</span>
          <span>0</span>
        </div>
        <div
          className="chart-plot"
          aria-label="Two-outcome market probability path"
          aria-describedby="market-path-description"
        >
          {[0.25, 0.5, 0.75].map((line) => (
            <i key={line} style={{ top: `${line * 100}%` }} />
          ))}
          {outcomeSeries.slice(0, 2).map((item, seriesIndex) => {
            const points = (item.points ?? []).slice(-80);
            return points.map((point: JsonRecord, index: number) => {
              if (!point.available || point.vwap === null) return null;
              const value = point.vwap;
              return (
                <b
                  key={`${seriesIndex}-${index}`}
                  className={`${seriesIndex ? "outcome-b" : "outcome-a"} ${
                    item.provenance === "derived_complement" ? "derived-point" : ""
                  }`}
                  style={{
                    left: `${(index / Math.max(1, points.length - 1)) * 100}%`,
                    top: `${(1 - Math.max(0, Math.min(1, value))) * 100}%`,
                  }}
                  title={`${item.outcome}: ${value.toFixed(3)}`}
                />
              );
            });
          })}
          {!market && (
            <div className="chart-empty">
              Select one logical market to load its source-time path.
            </div>
          )}
        </div>
        <div className="chart-legend">
          {outcomeSeries.slice(0, 2).map((item, index) => (
            <span key={`${item.outcome}-${index}`} className="series-legend">
              <span>
                <i className={index ? "legend-b" : "legend-a"} />
                {item.outcome} · {item.provenance.replaceAll("_", " ")}
              </span>
              {!!item.unavailablePointCount && (
                <em className="state-unavailable">
                  {item.unavailablePointCount} unavailable
                </em>
              )}
              {!!item.discontinuityCount && (
                <em className="state-ambiguous">
                  {item.discontinuityCount} discontinuities
                </em>
              )}
            </span>
          ))}
          {!outcomeSeries.length && (
            <>
              <span><i className="legend-a" />Outcome A · observed</span>
              <span><i className="legend-b" />Outcome B · derived / pending</span>
            </>
          )}
        </div>
      </div>
      {!!bboSeries.length && (
        <div className="bbo-chart">
          <div className="bbo-heading">
            <strong>Kalshi one-minute BBO OHLC</strong>
            <span>Bid and ask are source candles, not reconstructed quotes.</span>
          </div>
          {bboSeries.map((series) => {
            const points = series.points.slice(-60);
            return (
              <div className="bbo-outcome" key={series.outcome}>
                <div className="bbo-outcome-heading">
                  <strong>Outcome · {series.outcome}</strong>
                  <span>
                    {series.ambiguousPointCount
                      ? `${series.ambiguousPointCount} source-order ambiguous`
                      : series.unavailablePointCount
                        ? `${series.unavailablePointCount} unavailable`
                      : "complete OHLC"}
                    {" · "}
                    {series.discontinuityCount
                      ? `${series.discontinuityCount} discontinuities`
                      : "continuous segments"}
                  </span>
                </div>
                <div
                  className="bbo-plot"
                  aria-label={`Kalshi ${series.outcome} bid and ask OHLC candle path`}
                >
                  {points.map((point: JsonRecord, index: number) => {
                    const x =
                      (index / Math.max(1, points.length - 1)) * 100;
                    if (point.status === "ambiguous") {
                      return (
                        <i
                          className="bbo-ambiguous"
                          key={`${point.sourceStart}-${index}`}
                          style={{ left: `${x}%` }}
                          title="Source order ambiguous; no within-interval order inferred"
                        >
                          ?
                        </i>
                      );
                    }
                    if (!point.available) {
                      return (
                        <i
                          className="bbo-unavailable"
                          key={`${point.sourceStart}-${index}`}
                          style={{ left: `${x}%` }}
                          title="OHLC unavailable; no numeric value drawn"
                        >
                          ×
                        </i>
                      );
                    }
                    return (
                      <div
                        className="bbo-candle"
                        key={`${point.sourceStart}-${index}`}
                        style={{ left: `${x}%` }}
                      >
                        {point.discontinuous && (
                          <i
                            className="bbo-discontinuity"
                            title="Source segment discontinuity"
                          />
                        )}
                        <i
                          className="bbo-stick bid"
                          style={{
                            top: `${(1 - point.bid.high) * 100}%`,
                            height: `${Math.max(
                              0.5,
                              (point.bid.high - point.bid.low) * 100,
                            )}%`,
                          }}
                          title={`Bid O/H/L/C ${point.bid.open}/${point.bid.high}/${point.bid.low}/${point.bid.close}`}
                        />
                        <i
                          className="bbo-stick ask"
                          style={{
                            top: `${(1 - point.ask.high) * 100}%`,
                            height: `${Math.max(
                              0.5,
                              (point.ask.high - point.ask.low) * 100,
                            )}%`,
                          }}
                          title={`Ask O/H/L/C ${point.ask.open}/${point.ask.high}/${point.ask.low}/${point.ask.close}`}
                        />
                        <b className="bid-open" style={{ top: `${(1 - point.bid.open) * 100}%` }} />
                        <b className="bid-close" style={{ top: `${(1 - point.bid.close) * 100}%` }} />
                        <b className="ask-open" style={{ top: `${(1 - point.ask.open) * 100}%` }} />
                        <b className="ask-close" style={{ top: `${(1 - point.ask.close) * 100}%` }} />
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
          <div className="bbo-legend">
            <span><i className="bid" />Bid OHLC</span>
            <span><i className="ask" />Ask OHLC</span>
          </div>
        </div>
      )}
      <p id="market-path-description" className="sr-only">
        Source-time chart. Trade values are observed or explicitly marked as
        derived. Missing values are unavailable and discontinuities do not join
        across source segments. Kalshi BBO values are one-minute bid and ask
        OHLC candles, not reconstructed executable quotes. Source-order
        ambiguous intervals have no within-interval order inferred.
      </p>
      <div className="accessible-market-data">
        <table className="accessible-evidence-table">
          <caption>Trade range evidence</caption>
          <thead>
            <tr>
              <th>Source time</th>
              <th>Outcome</th>
              <th>Value</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {outcomeSeries.flatMap((series) =>
              series.points.slice(-80).map((point: JsonRecord, index: number) => (
                <tr key={`${series.outcome}-${point.source_start}-${index}`}>
                  <td>{point.source_start ?? "unavailable"}</td>
                  <td>{series.outcome}</td>
                  <td>{point.available ? point.vwap : "unavailable"}</td>
                  <td>
                    {point.discontinuous
                      ? "discontinuity"
                      : point.available
                        ? series.provenance
                        : "unavailable"}
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
        <table className="accessible-evidence-table">
          <caption>Kalshi BBO OHLC evidence</caption>
          <thead>
            <tr>
              <th>Source time</th>
              <th>Outcome</th>
              <th>Bid OHLC</th>
              <th>Ask OHLC</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {bboSeries.flatMap((series) =>
              series.points.slice(-60).map((point: JsonRecord, index: number) => (
                <tr key={`${series.outcome}-${point.sourceStart}-${index}`}>
                  <td>{point.sourceStart ?? "unavailable"}</td>
                  <td>{series.outcome}</td>
                  <td>{ohlcLabel(point.bid)}</td>
                  <td>{ohlcLabel(point.ask)}</td>
                  <td>
                    {point.status === "ambiguous"
                      ? "source order ambiguous"
                      : point.discontinuous
                        ? "discontinuity"
                      : point.available
                        ? "observed candle"
                        : "unavailable"}
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}

function StableId({ label, value }: { label: string; value?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="stable-id">
      <span>{label}</span>
      <code>{value ?? "unavailable"}</code>
      <button
        type="button"
        onClick={async () => {
          if (!value) return;
          await navigator.clipboard.writeText(value);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1200);
        }}
        disabled={!value}
      >
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}

export function DashboardWorkspace() {
  const client = useRef(new DashboardDataClient());
  const gameRequestGeneration = useRef(0);
  const marketRequestGeneration = useRef(0);
  const associationRequestGeneration = useRef(0);
  const contractRequestGeneration = useRef(0);
  const deepLinkReference = useRef<ReturnType<
    typeof parseDashboardReference
  > | null>(null);
  const gameAbortController = useRef<AbortController | null>(null);
  const marketAbortController = useRef<AbortController | null>(null);
  const associationAbortController = useRef<AbortController | null>(null);
  const contractAbortController = useRef<AbortController | null>(null);
  const [catalog, setCatalog] = useState<JsonRecord | null>(null);
  const [selectedGameId, setSelectedGameId] = useState("2025_14_DAL_DET");
  const [core, setCore] = useState<JsonRecord | null>(null);
  const [contracts, setContracts] = useState<JsonRecord[]>([]);
  const [marketSelection, dispatchMarketSelection] = useReducer(
    reduceMarketSelection,
    {
      id: "",
      market: null,
      error: "",
      loading: false,
      generation: 0,
    },
  );
  const selectedMarketId = marketSelection.id;
  const market = marketSelection.market;
  const [associations, setAssociations] = useState<JsonRecord[]>([]);
  const [eventIndex, setEventIndex] = useState(0);
  const [family, setFamily] = useState("all");
  const [venue, setVenue] = useState("all");
  const [marketState, setMarketState] = useState("analysis_eligible");
  const [delay, setDelay] = useState("0");
  const [horizon, setHorizon] = useState("10");
  const [includeAmbiguous, setIncludeAmbiguous] = useState(false);
  const [includeContaminated, setIncludeContaminated] = useState(false);
  const [associationStatus, setAssociationStatus] = useState("OBSERVED");
  const [auditOpen, setAuditOpen] = useState(false);
  const [busy, setBusy] = useState("Connecting to private evidence store…");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    const reference = parseDashboardReference(window.location.search);
    deepLinkReference.current = reference;
    client.current
      .loadCatalog()
      .then((value: JsonRecord) => {
        if (!active) return;
        setCatalog(value);
        setDelay(reference.delay);
        setHorizon(reference.horizon);
        setBusy("");
        return openGame(reference.gameId ?? selectedGameId, value);
      })
      .catch((caught: Error) => {
        if (!active) return;
        setError(caught.message);
        setBusy("");
      });
    return () => {
      active = false;
      gameAbortController.current?.abort();
      marketAbortController.current?.abort();
      associationAbortController.current?.abort();
      contractAbortController.current?.abort();
    };
    // The initial source-of-truth fetch is intentionally once per session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function openGame(gameId: string, knownCatalog = catalog) {
    if (!knownCatalog) return;
    const generation = ++gameRequestGeneration.current;
    gameAbortController.current?.abort();
    marketAbortController.current?.abort();
    associationAbortController.current?.abort();
    contractAbortController.current?.abort();
    ++marketRequestGeneration.current;
    ++associationRequestGeneration.current;
    ++contractRequestGeneration.current;
    const controller = new AbortController();
    gameAbortController.current = controller;
    setSelectedGameId(gameId);
    setBusy(`Loading ${gameId} replay…`);
    setError("");
    setCore(null);
    setContracts([]);
    dispatchMarketSelection({
      type: "clear",
      generation: marketRequestGeneration.current,
    });
    setAssociations([]);
    setEventIndex(0);
    try {
      const value = await client.current.selectGame(gameId, {
        signal: controller.signal,
      });
      if (
        controller.signal.aborted ||
        generation !== gameRequestGeneration.current
      ) {
        return;
      }
      setCore(value);
      const reference = deepLinkReference.current;
      if (reference?.gameId === gameId || (!reference?.gameId && generation === 1)) {
        if (reference.episodeId) {
          const referencedEventIndex = resolveEpisodeEventIndex(
            value,
            reference.episodeId,
          );
          if (referencedEventIndex === null) {
            setError(
              `Deep-link episode is not present in the verified game projection: ${reference.episodeId}`,
            );
          } else {
            setEventIndex(referencedEventIndex);
          }
        }
        if (reference.marketId) {
          const contractValue = await client.current.loadContracts({
            signal: controller.signal,
          });
          if (
            controller.signal.aborted ||
            generation !== gameRequestGeneration.current
          ) {
            return;
          }
          const contractRows = contractValue.payload.contracts ?? [];
          setContracts(contractRows);
          const marketReference = resolveMarketReference(
            contractRows,
            reference.marketId,
          );
          if (!marketReference) {
            setError(
              `Deep-link market is not present in the verified game projection: ${reference.marketId}`,
            );
          } else {
            setFamily(marketReference.family);
            setVenue(marketReference.venue);
            setMarketState(marketReference.marketState);
            await openMarket(marketReference.logicalMarketId);
          }
        }
        deepLinkReference.current = null;
      }
    } catch (caught) {
      if (
        !controller.signal.aborted &&
        generation === gameRequestGeneration.current &&
        !/superseded/i.test((caught as Error).message)
      ) {
        setError((caught as Error).message);
      }
    } finally {
      if (generation === gameRequestGeneration.current) setBusy("");
    }
  }

  async function openContracts() {
    const generation = ++contractRequestGeneration.current;
    contractAbortController.current?.abort();
    const controller = new AbortController();
    contractAbortController.current = controller;
    setBusy("Loading contract inventory…");
    try {
      const value = await client.current.loadContracts({
        signal: controller.signal,
      });
      if (
        controller.signal.aborted ||
        generation !== contractRequestGeneration.current
      ) {
        return;
      }
      setContracts(value.payload.contracts ?? []);
    } catch (caught) {
      if (
        !controller.signal.aborted &&
        generation === contractRequestGeneration.current
      ) {
        setError((caught as Error).message);
      }
    } finally {
      if (generation === contractRequestGeneration.current) setBusy("");
    }
  }

  async function openMarket(logicalMarketId: string) {
    const generation = ++marketRequestGeneration.current;
    marketAbortController.current?.abort();
    if (!logicalMarketId) {
      dispatchMarketSelection({ type: "clear", generation });
      setBusy("");
      return;
    }
    const controller = new AbortController();
    marketAbortController.current = controller;
    dispatchMarketSelection({
      type: "request",
      id: logicalMarketId,
      generation,
    });
    setBusy("Loading selected market path…");
    try {
      const value = await client.current.selectMarket(logicalMarketId, {
        signal: controller.signal,
      });
      if (
        controller.signal.aborted ||
        generation !== marketRequestGeneration.current
      ) {
        return;
      }
      dispatchMarketSelection({
        type: "success",
        id: logicalMarketId,
        generation,
        market: value,
      });
    } catch (caught) {
      if (
        !controller.signal.aborted &&
        generation === marketRequestGeneration.current &&
        !/superseded/i.test((caught as Error).message)
      ) {
        dispatchMarketSelection({
          type: "failure",
          id: logicalMarketId,
          generation,
          error: (caught as Error).message,
        });
      }
    } finally {
      if (generation === marketRequestGeneration.current) setBusy("");
    }
  }

  async function openAssociations() {
    const generation = ++associationRequestGeneration.current;
    associationAbortController.current?.abort();
    const controller = new AbortController();
    associationAbortController.current = controller;
    setBusy("Loading bounded association preview…");
    try {
      const value = await client.current.loadAssociations({
        signal: controller.signal,
      });
      if (
        controller.signal.aborted ||
        generation !== associationRequestGeneration.current
      ) {
        return;
      }
      setAssociations(value.payload.association_preview ?? []);
    } catch (caught) {
      if (
        !controller.signal.aborted &&
        generation === associationRequestGeneration.current &&
        !/superseded/i.test((caught as Error).message)
      ) {
        setError((caught as Error).message);
      }
    } finally {
      if (generation === associationRequestGeneration.current) setBusy("");
    }
  }

  function updateContractFilter(
    key: "family" | "venue" | "state",
    value: string,
  ) {
    const nextFilters = {
      family,
      venue,
      state: marketState,
      [key]: value,
    };
    const nextContracts = filterContracts(contracts, nextFilters);
    const generation = marketRequestGeneration.current + 1;
    if (
      reconcileMarketSelection(
        marketSelection,
        nextContracts,
        generation,
      ) !== marketSelection
    ) {
      marketAbortController.current?.abort();
      marketRequestGeneration.current = generation;
      dispatchMarketSelection({ type: "clear", generation });
      setBusy("");
    }
    if (key === "family") setFamily(value);
    if (key === "venue") setVenue(value);
    if (key === "state") setMarketState(value);
  }

  const games: JsonRecord[] = catalog
    ? buildGameIndexRows(catalog)
    : frozenGames.map((gameId) => ({
        gameId,
        awayTeam: gameTeams(gameId).away,
        homeTeam: gameTeams(gameId).home,
        venues: [],
        auditGate: "AWAITING_PRIVATE_CATALOG",
        status: "PRELIMINARY_SOURCE_TIME_ONLY",
      }));
  const selectedSummary =
    games.find((game) => game.gameId === selectedGameId) ?? games[0];
  const events: JsonRecord[] = core?.payload?.events ?? [];
  const replayEvidence = replayEvidenceAt(core, eventIndex);
  const currentEvent = replayEvidence.event;
  const currentEpisodeId = episodeIdAtEvent(core, eventIndex);
  const families = [...new Set(contracts.map((item) => item.family))].sort();
  const venues = [...new Set(contracts.map((item) => item.venue))].sort();
  const associationStatuses = [
    ...new Set(associations.map((item) => item.validity_status)),
  ].sort();
  const filteredContracts = useMemo(
    () =>
      filterContracts(contracts, {
        family,
        venue,
        state: marketState,
      }),
    [contracts, family, venue, marketState],
  );
  const visibleAssociations = useMemo(
    () =>
      filterAssociationRows(associations, {
        delay,
        horizon,
        includeAmbiguous,
        includeContaminated,
        status: associationStatus,
      }).slice(0, 12),
    [
      associations,
      delay,
      horizon,
      includeAmbiguous,
      includeContaminated,
      associationStatus,
    ],
  );
  const teams = core?.payload?.teams ?? gameTeams(selectedGameId);
  const lineage = lineageEntries(core);

  return (
    <main className="workspace">
      <header className="masthead">
        <div className="wordmark">
          <span>SAF</span>
          <small>Sports Alpha Finding</small>
        </div>
        <div className="research-title">
          <p>X-13 · NFL 2025 · source-time evidence</p>
          <h1>Game State × Market Reaction</h1>
        </div>
        <div className="boundary">
          <strong>PRELIMINARY_SOURCE_TIME_ONLY</strong>
          <span>No causality · no real latency · no execution claim</span>
        </div>
      </header>

      {(busy || error || marketSelection.error) && (
        <div
          className={`system-message ${error || marketSelection.error ? "error" : ""}`}
          role="status"
        >
          {error || marketSelection.error || busy}
        </div>
      )}

      <section className="work-grid">
        <aside className="game-index" aria-label="Game index">
          <div className="section-label">
            <span>01</span>
            <h2>20-game evidence index</h2>
          </div>
          <p className="index-note">Frozen before market-direction review</p>
          <div className="game-list">
            {games.map((game, index) => {
              const teamsForGame = {
                away: game.awayTeam,
                home: game.homeTeam,
              };
              return (
                <button
                  key={game.gameId}
                  type="button"
                  className={game.gameId === selectedGameId ? "active" : ""}
                  onClick={() => openGame(game.gameId)}
                  disabled={!catalog}
                  aria-pressed={game.gameId === selectedGameId}
                >
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <strong>
                    {teamsForGame.away} <i>at</i> {teamsForGame.home}
                  </strong>
                  <em>
                    {game.awayScore ?? "—"}–{game.homeScore ?? "—"}
                  </em>
                  <small>
                    {game.eventCount ?? "—"} plays · {game.episodeCount ?? "—"} episodes
                  </small>
                  <small className="game-market-meta">
                    {game.contractCount ?? "—"} markets ·{" "}
                    {game.venues.length ? game.venues.join(" + ") : "venue pending"}
                  </small>
                    <small className={evidenceStateClass(game.auditGate)}>
                    {game.auditGate}
                  </small>
                </button>
              );
            })}
          </div>
        </aside>

        <div className="replay-column">
          <section className="scoreboard">
            <div className="team away">
              <span>AWAY</span>
              <strong>{teams.away ?? selectedSummary?.awayTeam ?? "—"}</strong>
              <b>{currentEvent?.away_score ?? selectedSummary?.awayScore ?? "—"}</b>
            </div>
            <div className="game-clock">
              <span>{clockLabel(currentEvent)}</span>
              <strong>
                {currentEvent?.down
                  ? `${currentEvent.down}${["th", "st", "nd", "rd"][currentEvent.down] ?? "th"} & ${currentEvent.distance ?? "—"}`
                  : "BETWEEN PLAYS"}
              </strong>
              <small>{shortTime(currentEvent?.source_time_start_utc)}</small>
            </div>
            <div className="team home">
              <span>HOME</span>
              <strong>{teams.home ?? selectedSummary?.homeTeam ?? "—"}</strong>
              <b>{currentEvent?.home_score ?? selectedSummary?.homeScore ?? "—"}</b>
            </div>
          </section>

          <section className="replay-panel">
            <div className="section-label">
              <span>02</span>
              <h2>Source-time game replay</h2>
              <em className={evidenceStateClass("OBSERVED")}>OBSERVED GAME SOURCE</em>
            </div>
            <FootballField event={currentEvent} />
            <div className="play-scrubber">
              <label htmlFor="play-range">
                Play scrubber <strong>{events.length ? eventIndex + 1 : 0} / {events.length}</strong>
              </label>
              <input
                id="play-range"
                type="range"
                min="0"
                max={Math.max(0, events.length - 1)}
                value={Math.min(eventIndex, Math.max(0, events.length - 1))}
                onChange={(event) => setEventIndex(Number(event.target.value))}
                disabled={!events.length}
              />
              <div className="scrubber-actions">
                <button
                  type="button"
                  onClick={() => setEventIndex((value) => Math.max(0, value - 1))}
                  disabled={eventIndex === 0}
                >
                  ← Previous
                </button>
                <button
                  type="button"
                  onClick={() =>
                    setEventIndex((value) => Math.min(events.length - 1, value + 1))
                  }
                  disabled={!events.length || eventIndex >= events.length - 1}
                >
                  Next →
                </button>
              </div>
            </div>
            <article className="current-play">
              <div>
                <span>PLAY {currentEvent?.play_id ?? "—"}</span>
                <h3>{currentEvent?.description ?? "Select a verified game to inspect its play sequence."}</h3>
                <EventPill event={currentEvent ?? {}} />
              </div>
              <dl>
                <div><dt>Possession</dt><dd>{currentEvent?.possession_team ?? "—"}</dd></div>
                <div><dt>Drive</dt><dd>{currentEvent?.drive_id ?? "—"}</dd></div>
                <div><dt>Formation</dt><dd>{currentEvent?.formation ?? "unavailable"}</dd></div>
                <div><dt>Personnel coverage</dt><dd>{replayEvidence.personnel.coverage}</dd></div>
              </dl>
            </article>
            <div className="personnel-coverage">
              <span className={evidenceStateClass(replayEvidence.personnel.coverage)}>
                {replayEvidence.personnel.coverage}
              </span>
              <p>{replayEvidence.personnel.semantics}</p>
              <strong>
                {replayEvidence.personnel.completeRows} complete 11v11 rows ·{" "}
                {replayEvidence.personnel.missingRows} missing rows
              </strong>
            </div>
            <div className="personnel-ledger">
              <div>
                <span>OFFENSE · {replayEvidence.offense.team ?? "—"}</span>
                <strong>{replayEvidence.offense.package ?? "package unavailable"}</strong>
                <p>{replayEvidence.offense.names.join(" · ") || "On-field names unavailable for this row."}</p>
              </div>
              <div>
                <span>DEFENSE · {replayEvidence.defense.team ?? "—"}</span>
                <strong>{replayEvidence.defense.package ?? "package unavailable"}</strong>
                <p>{replayEvidence.defense.names.join(" · ") || "On-field names unavailable for this row."}</p>
              </div>
            </div>
            <div className="cumulative-ledger">
              <div className="ledger-heading">
                <span>CUMULATIVE STATE</span>
                <strong>through play {replayEvidence.ledger.throughPlayId ?? "—"}</strong>
              </div>
              <div className="team-score-ledger">
                {Object.entries(replayEvidence.ledger.teamScores).map(([team, score]) => (
                  <span key={team}>{team} <strong>{String(score)}</strong></span>
                ))}
                {!Object.keys(replayEvidence.ledger.teamScores).length && <span>Ledger unavailable</span>}
              </div>
              <div className="player-stat-ledger">
                {replayEvidence.ledger.rolePlayerStats.map((row: JsonRecord) => (
                  <article key={`${row.role}-${row.player}`}>
                    <span>{row.role}</span>
                    <strong>{row.player}</strong>
                    <small>
                      PASS {row.passing_yards ?? 0}y · RUSH {row.rushing_yards ?? 0}y ·
                      REC {row.receiving_yards ?? 0}y
                    </small>
                    <small className="touchdown-ledger">
                      TD pass {row.passing_touchdowns ?? 0} · rush{" "}
                      {row.rushing_touchdowns ?? 0} · receiving{" "}
                      {row.receiving_touchdowns ?? 0} · total{" "}
                      {row.canonicalTouchdowns}
                    </small>
                  </article>
                ))}
                {!replayEvidence.ledger.rolePlayerStats.length && (
                  <p className="empty-copy">No role-linked cumulative player state on this row.</p>
                )}
              </div>
            </div>
          </section>

          <section className="event-tape">
            <div className="section-label">
              <span>03</span>
              <h2>Event tape</h2>
              <em>{events.length} source rows</em>
            </div>
            <div className="event-list">
              {events.slice(Math.max(0, eventIndex - 2), eventIndex + 4).map((event, offset) => {
                const absoluteIndex = Math.max(0, eventIndex - 2) + offset;
                return (
                  <button
                    key={`${event.play_id}-${absoluteIndex}`}
                    type="button"
                    className={absoluteIndex === eventIndex ? "active" : ""}
                    onClick={() => setEventIndex(absoluteIndex)}
                  >
                    <time>{shortTime(event.source_time_start_utc)}</time>
                    <span>{clockLabel(event)}</span>
                    <strong>{event.play_family}</strong>
                    <p>{event.description}</p>
                  </button>
                );
              })}
              {!events.length && <p className="empty-copy">Replay rows appear after a verified core asset loads.</p>}
            </div>
          </section>
        </div>

        <aside className="market-column">
          <section className="market-panel">
            <div className="section-label">
              <span>04</span>
              <h2>Market source path</h2>
            </div>
            <div className="market-controls">
              <label>
                Venue
                <select
                      value={venue}
                      onChange={(event) =>
                        updateContractFilter("venue", event.target.value)
                      }
                  disabled={!contracts.length}
                >
                  <option value="all">All venues</option>
                  {venues.map((value) => <option key={value}>{value}</option>)}
                </select>
              </label>
              <label>
                Market family
                <select
                  value={family}
                  onChange={(event) =>
                    updateContractFilter("family", event.target.value)
                  }
                  disabled={!contracts.length}
                >
                  <option value="all">All families</option>
                  {families.map((value) => <option key={value}>{value}</option>)}
                </select>
              </label>
              <label>
                Market state
                <select
                  value={marketState}
                  onChange={(event) =>
                    updateContractFilter("state", event.target.value)
                  }
                  disabled={!contracts.length}
                >
                  <option value="analysis_eligible">Analysis eligible</option>
                  <option value="excluded">Excluded / inventory only</option>
                  <option value="all">All states</option>
                </select>
              </label>
              <label>
                Logical market
                <select
                  value={selectedMarketId}
                  onChange={(event) => openMarket(event.target.value)}
                  disabled={!contracts.length}
                >
                  <option value="">Select a market</option>
                  {filteredContracts.map((contract: JsonRecord) => (
                    <option key={contract.logical_market_id} value={contract.logical_market_id}>
                      {contract.selector_label}
                    </option>
                  ))}
                </select>
              </label>
              {!contracts.length && (
                <button type="button" className="load-button" onClick={openContracts} disabled={!core}>
                  Load contract inventory
                </button>
              )}
            </div>
            <div className="semantic-key">
              <span className="state-observed">OBSERVED</span>
              <span className="state-derived">DERIVED</span>
              <span className="state-ambiguous">AMBIGUOUS</span>
              <span className="state-unavailable">UNAVAILABLE</span>
            </div>
            <MarketPath market={market} />
            <div className="bbo-band">
              <strong>Historical bid / ask</strong>
              <span>
                {market?.payload?.series?.some((item: JsonRecord) =>
                  String(item.series_kind).includes("bbo"),
                )
                  ? "Kalshi 1-minute BBO interval available"
                  : "Unavailable for selected market · no midpoint invented"}
              </span>
            </div>
          </section>

          <section className="association-panel">
            <div className="section-label">
              <span>05</span>
              <h2>Association query</h2>
            </div>
            <div className="filter-grid">
              <label>
                Delay scenario
                <select value={delay} onChange={(event) => setDelay(event.target.value)}>
                  {[0, 1, 2, 3, 5, 10].map((value) => <option key={value} value={value}>{value}s</option>)}
                </select>
              </label>
              <label>
                Horizon
                <select value={horizon} onChange={(event) => setHorizon(event.target.value)}>
                  {[1, 2, 5, 10, 30, 60].map((value) => <option key={value} value={value}>{value}s</option>)}
                </select>
              </label>
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={includeAmbiguous}
                  onChange={(event) => setIncludeAmbiguous(event.target.checked)}
                />
                Include ambiguous
              </label>
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={includeContaminated}
                  onChange={(event) => setIncludeContaminated(event.target.checked)}
                />
                Include contaminated
              </label>
              <label>
                Status
                <select
                  value={associationStatus}
                  onChange={(event) => setAssociationStatus(event.target.value)}
                >
                  <option value="OBSERVED">Observed / unflagged</option>
                  {associationStatuses
                    .filter((value) => value !== "OBSERVED")
                    .map((value) => <option key={value} value={value}>{value}</option>)}
                  <option value="all">All preview rows</option>
                </select>
              </label>
            </div>
            {!associations.length && (
              <button type="button" className="load-button" onClick={openAssociations} disabled={!core}>
                Load bounded association preview
              </button>
            )}
            <div className="association-results">
              {visibleAssociations.map((row: JsonRecord, index: number) => (
                <article key={`${row.episode_id}-${row.logical_market_id}-${index}`}>
                  <span className={evidenceStateClass(row.validity_status)}>{row.validity_status}</span>
                  <strong>{row.episode_type} → {row.outcome}</strong>
                  <em>{row.signed_price_change ?? "no paired change"}</em>
                  <small>{row.trade_count} trades · {row.volume ?? "—"} volume</small>
                </article>
              ))}
              {!!associations.length && !visibleAssociations.length && (
                <p className="empty-copy">No preview rows satisfy the current evidence filters.</p>
              )}
            </div>
          </section>

          <section className={`audit-drawer ${auditOpen ? "open" : ""}`}>
            <button type="button" onClick={() => setAuditOpen((value) => !value)} aria-expanded={auditOpen}>
              <span>Audit & lineage</span>
              <strong>{auditOpen ? "Close −" : "Inspect +"}</strong>
            </button>
            {auditOpen && (
              <div className="audit-content">
                <div>
                  {Object.entries(core?.payload?.audit ?? {
                    replay: "awaiting verified asset",
                    layer_g: "awaiting verified asset",
                    layer_m: "awaiting verified asset",
                  }).map(([key, value]) => (
                    <p key={key}><span>{key.replaceAll("_", " ")}</span><strong>{String(value)}</strong></p>
                  ))}
                </div>
                {lineage.map((row) => (
                  <StableId key={row.label} label={row.label} value={row.value} />
                ))}
                <StableId label="Batch ID" value={catalog?.batch_id} />
                <StableId label="Game ID" value={selectedGameId} />
                <StableId label="Episode ID" value={currentEpisodeId ?? undefined} />
                <StableId label="Play ID" value={currentEvent?.play_id} />
                <StableId label="Market ID" value={selectedMarketId} />
              </div>
            )}
          </section>
        </aside>
      </section>

      <footer>
        <span>SAF research surface · human review before promotion</span>
        <strong>Observed ≠ executable · source time ≠ receive time</strong>
      </footer>
    </main>
  );
}
