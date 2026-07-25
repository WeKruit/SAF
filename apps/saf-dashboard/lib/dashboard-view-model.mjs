export function buildGameIndexRows(catalog) {
  return (catalog?.payload?.games ?? []).map((game) => ({
    gameId: game.game_id,
    awayTeam: game.away_team,
    homeTeam: game.home_team,
    awayScore: game.away_score,
    homeScore: game.home_score,
    eventCount: game.event_count,
    episodeCount: game.episode_count,
    contractCount: game.contract_count,
    venues: Array.isArray(game.venues) ? game.venues : [],
    auditGate: game.audit?.layer_a ?? "NOT_AUDITED",
    status: game.status,
  }));
}

export function evidenceStateClass(value) {
  const state = typeof value === "string" ? value.toUpperCase() : "";
  if (
    new Set([
      "OBSERVED",
      "PASS",
      "PASS_SOURCE_TIME_ONLY",
      "PASS_RESEARCH_ONLY",
      "VERIFIED",
    ]).has(state)
  ) {
    return "state-observed";
  }
  if (
    new Set([
      "ORDER_AMBIGUOUS",
      "CONTAMINATED",
      "RESEARCH_ONLY",
      "PRELIMINARY_SOURCE_TIME_ONLY",
    ]).has(state)
  ) {
    return state === "CONTAMINATED"
      ? "state-contaminated"
      : "state-ambiguous";
  }
  return "state-unavailable";
}

function matchingLedger(payload, event) {
  const ledger = payload?.stat_ledger ?? [];
  const exact = ledger.find(
    (row) => String(row.through_play_id) === String(event?.play_id),
  );
  if (exact) return exact;
  const sequence = Number(event?.order_sequence);
  if (!Number.isFinite(sequence)) return null;
  return (
    ledger
      .filter((row) => Number(row.through_order_sequence) <= sequence)
      .sort(
        (left, right) =>
          Number(right.through_order_sequence) -
          Number(left.through_order_sequence),
      )[0] ?? null
  );
}

export function replayEvidenceAt(core, eventIndex) {
  const payload = core?.payload ?? {};
  const event = payload.events?.[eventIndex] ?? null;
  const sourcePersonnel = payload.personnel ?? {};
  const ledger = matchingLedger(payload, event);
  const roles = Object.entries(event?.player_roles ?? {});
  const rolePlayerStats = roles
    .map(([role, player]) => {
      const stats = ledger?.player_stats?.[player];
      if (!stats) return null;
      const canonicalTouchdowns = [
        stats.passing_touchdowns,
        stats.rushing_touchdowns,
        stats.receiving_touchdowns,
      ].reduce(
        (total, value) =>
          total + (Number.isFinite(Number(value)) ? Number(value) : 0),
        0,
      );
      return { role, player, ...stats, canonicalTouchdowns };
    })
    .filter(Boolean);
  return {
    event,
    personnel: {
      coverage: sourcePersonnel.coverage ?? "unavailable",
      licenseStatus: sourcePersonnel.license_status ?? "unavailable",
      completeRows: sourcePersonnel.complete_11v11_rows ?? 0,
      missingRows: sourcePersonnel.missing_rows ?? 0,
      semantics: sourcePersonnel.semantics ?? "unavailable",
    },
    offense: {
      team: event?.offense_team ?? null,
      package: event?.offense_personnel ?? null,
      names: Array.isArray(event?.offense_names) ? event.offense_names : [],
    },
    defense: {
      team: event?.defense_team ?? null,
      package: event?.defense_personnel ?? null,
      names: Array.isArray(event?.defense_names) ? event.defense_names : [],
    },
    ledger: {
      throughPlayId: ledger?.through_play_id ?? null,
      teamScores: ledger?.team_scores ?? {},
      periodScores: ledger?.period_scores ?? {},
      rolePlayerStats,
    },
  };
}

function nullableNumber(value) {
  if (
    value === null ||
    value === undefined ||
    (typeof value === "string" && value.trim() === "")
  ) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function numericOhlc(value) {
  const result = {};
  if (!value || typeof value !== "object") {
    for (const field of ["open", "high", "low", "close"]) {
      result[field] = null;
    }
    return result;
  }
  for (const field of ["open", "high", "low", "close"]) {
    result[field] = nullableNumber(value[field]);
  }
  return result;
}

export function reduceMarketSelection(state, action) {
  if (!action || action.generation < state.generation) return state;
  if (action.type === "clear") {
    return {
      id: "",
      market: null,
      error: "",
      loading: false,
      generation: action.generation,
    };
  }
  if (action.type === "request") {
    return {
      id: action.id,
      market: null,
      error: "",
      loading: true,
      generation: action.generation,
    };
  }
  if (
    (action.type === "success" || action.type === "failure") &&
    action.id !== state.id
  ) {
    return state;
  }
  if (action.type === "success") {
    return {
      id: action.id,
      market: action.market,
      error: "",
      loading: false,
      generation: action.generation,
    };
  }
  if (action.type === "failure") {
    return {
      id: action.id,
      market: null,
      error: action.error,
      loading: false,
      generation: action.generation,
    };
  }
  return state;
}

export function reconcileMarketSelection(
  state,
  filteredContracts,
  generation,
) {
  if (
    !state.id ||
    filteredContracts.some(
      (contract) => contract.logical_market_id === state.id,
    )
  ) {
    return state;
  }
  return reduceMarketSelection(state, { type: "clear", generation });
}

function bboPointStatus(point) {
  if (point.sourceOrderAmbiguous) return "ambiguous";
  if (!point.available) return "unavailable";
  if (point.discontinuous) return "discontinuity";
  return "observed";
}

export function marketSeriesView(market) {
  const series = market?.payload?.series ?? [];
  const trades = series
    .filter((item) => item.series_kind === "trade_range")
    .map((item) => {
      const points = (item.points ?? []).map((point) => {
        const vwap = nullableNumber(point.vwap);
        return {
          ...point,
          vwap,
          available: vwap !== null,
          discontinuous: point.contains_discontinuity === true,
        };
      });
      return {
        ...item,
        points,
        unavailablePointCount: points.filter((point) => !point.available)
          .length,
        discontinuityCount: points.filter((point) => point.discontinuous)
          .length,
      };
    });
  const bbo = series
    .filter((item) => item.series_kind === "bbo_candle")
    .map((item) => {
      const points = (item.points ?? []).map((point) => {
          const bid = numericOhlc(point.bid_ohlc);
          const ask = numericOhlc(point.ask_ohlc);
          const available = [...Object.values(bid), ...Object.values(ask)].every(
            (value) => value !== null,
          );
          const normalizedPoint = {
            bid,
            ask,
            available,
            discontinuous: point.contains_discontinuity === true,
            sourceStart: point.source_start,
            sourceEnd: point.source_end,
            sourceOrderAmbiguous: point.source_order_ambiguous === true,
          };
          return {
            ...normalizedPoint,
            status: bboPointStatus(normalizedPoint),
          };
        });
      return {
        outcome: item.outcome,
        provenance: item.provenance,
        points,
        unavailablePointCount: points.filter((point) => !point.available)
          .length,
        discontinuityCount: points.filter((point) => point.discontinuous)
          .length,
        ambiguousPointCount: points.filter(
          (point) => point.status === "ambiguous",
        ).length,
      };
    });
  return { trades, bbo };
}

export function filterContracts(contracts, filters) {
  return contracts.filter(
    (contract) =>
      (filters.family === "all" || contract.family === filters.family) &&
      (filters.venue === "all" || contract.venue === filters.venue) &&
      (filters.state === "all" ||
        (filters.state === "analysis_eligible" &&
          contract.analysis_eligible === true) ||
        (filters.state === "excluded" && contract.analysis_eligible !== true)),
  );
}

export function filterAssociationRows(rows, filters) {
  return rows.filter(
    (row) =>
      String(row.delay_scenario_seconds) === filters.delay &&
      String(row.horizon_seconds) === filters.horizon &&
      (filters.includeAmbiguous || !row.order_ambiguous) &&
      (filters.includeContaminated || !row.contaminated) &&
      (filters.status === "all" || row.validity_status === filters.status),
  );
}

export function lineageEntries(core) {
  const lineage = core?.payload?.lineage ?? {};
  const hashes = Array.isArray(lineage.raw_manifest_hashes)
    ? lineage.raw_manifest_hashes
    : [];
  return [
    { label: "Builder", value: lineage.builder_version ?? "unavailable" },
    { label: "Analysis lock", value: lineage.analysis_lock ?? "unavailable" },
    {
      label: "Game-state ledger",
      value: lineage.game_state_ledger_sha256 ?? "unavailable",
    },
    { label: "Raw manifests", value: `${hashes.length} verified hashes` },
    ...hashes.slice(0, 3).map((value, index) => ({
      label: `Raw hash ${index + 1}`,
      value,
    })),
  ];
}
