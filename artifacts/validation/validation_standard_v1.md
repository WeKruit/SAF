# Validation Standard v1

- **Owner:** Team H
- **Version:** v1
- **Program state:** `CONDITIONAL_GO`
- **Source of truth:** `charter/research_program_charter_v0.2.md`, `catalog_registry.csv`, `catalog_team_assignments.csv`
- **Current experiment inventory:** X-01 through X-13

Validation Standard v0 remains an immutable historical artifact. This v1
supersedes it as the current validation standard and carries forward every v0
registration, leakage, deterministic replay, compliance, promotion, and NO-GO
rule without relaxing any gate.

## X-13 registration boundary

X-13 is the only newly authorized experiment. Its scope is exactly the frozen
twenty-game NFL 2025 source-time replay and historical Polymarket/Kalshi
exploration. The fixed sample, venue bindings, contract universe, timestamp
interval semantics, delay scenarios, horizons, event taxonomy, minimum counts,
and analysis whitelist were registered before inspecting market price
direction.

Only `preliminary_source_time_only` is authorized. The raw source-manifest
bundle must be content-addressed and resolved by a Team H amendment before any
association result is evaluated. The following remain invalid:

1. a result from an unregistered game, contract family, horizon, interaction,
   dataset, model, or analysis scope;
2. a result missing terminal pagination proof, immutable raw manifests, code
   hash, input hash, or the controlling registration head;
3. a causal, true-latency, lead-lag, executable-price, fill, alpha, or trading
   claim derived from historical source timestamps;
4. treating a same-second source event and market observation as ordered;
5. carrying a stale trade forward to fill a missing horizon;
6. treating a cross-game MVE as a single-game contract;
7. constructing historical bid/ask or L2 from trades or midpoint;
8. treating targeted rare-event enrichment as league-representative incidence.

## X-13 replay and time gates

Game and market layers are validated independently before association:

- Game seconds are intervals. Market trades are one-second intervals. Kalshi
  candles are one-minute intervals.
- Delay values `0/1/2/3/5/10s` are scenarios, not measured latency.
- Horizons are exactly `1/2/5/10/30/60s`.
- Overlapping intervals are `order_ambiguous`; the next salient episode makes
  the window `contaminated`.
- Deleted plays are audit-only. No-play scores and turnovers remain nullified.
  TD tries and intervening administrative rows belong to one finalized episode.
- Personnel rows establish observed on-field sets only; they do not establish
  exact substitution or injury time.
- Every canonical artifact is rebuilt twice. A different event order or hash
  fails deterministic replay.

## Reporting gates

Primary pooled estimates require at least 300 eligible episodes across 16 games
and 100 observations per venue. Binary moderators require 40 episodes on each
side across eight games. Named subtypes require 20 episodes across six games;
sequences require 15 across six games. Five episodes across three games are
descriptive only. Everything smaller is `CASE_ONLY`. A single game contributing
more than 25 percent of a category downgrades that category to descriptive.

Game-cluster bootstrap, leave-one-game-out stability, and multiple-testing
control are mandatory for qualifying estimates. Rare safeties, onside
recoveries, blocked punts, and defensive two-point conversions remain case
series unless their frozen count gate is met.

## Unchanged program NO-GO

No real-money execution, live maker, exact PMXT queue-fill claim, multi-venue
live arbitrage, live copy trading, LLM hot path, reinforcement learning,
large-scale microservices, unregistered backtest, or README-return evidence is
authorized. Team I green remains a prerequisite for any production use.
