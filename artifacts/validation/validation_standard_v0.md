# Validation Standard v0

- **Owner:** Team H
- **Version:** v0
- **Program state:** `CONDITIONAL_GO`
- **Source of truth:** `charter/research_program_charter_v0.2.md`, `catalog_registry.csv`, `catalog_team_assignments.csv`
- **Due gate:** `first_round_artifact_gate`

This standard operationalizes the Charter. It does not alter approved scope, promotion gates, or NO-GO decisions.

## 1. Registration is a hard prerequisite

Only X-01 through X-10 exist in registry v0. A result is invalid when its experiment, result scope, code hash, input-data hash, or result-artifact hash was not registered before evaluation began. An unknown experiment remains invalid even if it supplies a syntactically valid hash. “Quick backtests” outside the registry have no evidentiary value.

The base card is content-addressed and immutable. Corrections are append-only amendments with a contiguous sequence, the preceding record hash, an amendment content hash, an H approver, time, reason, and explicit changes. `failed` and `abandoned` records are retained permanently; history is never overwritten.

Because the source program registration is dated but not time-stamped, v0 conservatively accepts evaluation only from `2026-07-23T00:00:00Z`. This is an evidence-acceptance boundary, not a claim that registration occurred at an invented time.

## 2. Authorization is scope-specific and fail-closed

`execution_authorized` describes acceptance of the complete/formal card. A true sub-scope never authorizes another sub-scope. Dependencies and every scope-specific registration lock must also be complete.

- X-05 authorizes spec drafting only; label generation and a formal result remain blocked until U/L/h, purge, embargo, overlap, suspension/resume, same-time-touch, and H-signature locks clear.
- X-07 authorizes only a result labeled `PRELIMINARY`; formal conclusions and go/no-go use remain blocked until point-in-time `venue_rule_snapshot` data are consumed by replay.
- X-08 authorizes the stopped-archive audit and public Polymarket capture. Kalshi capture remains credential-blocked, and the dual-venue result requires seven actual elapsed days. Fixtures cannot satisfy elapsed time. The Charter's `99% <= uptime < 100%` band is unresolved and is never silently passed.
- X-10 authorizes the precision audit only after its sample and Router/taxonomy locks clear. Recall requires a registered candidate-universe denominator. Live multi-venue arbitrage remains a permanent NO-GO regardless of precision.

## 3. Point-in-time and leakage discipline

Every input declares its source, version or explicit unresolved version lock, and point-in-time basis. Raw observations are append-only and hashed; normalized data must be reconstructible from raw data. Present-day venue rules must never backfill historical simulation: replay joins the versioned rule stream as of simulated time.

The default predictive split is game-grouped, chronological, and walk-forward. Any `n.a.` audit split requires an H approval lock. No game may cross folds. Purge and embargo parameters are registered before inspection. X-04 treats sports event time as an interval, not a falsely precise production timestamp.

Formal labels and execution evidence use executable ask for entry and bid for exit. Midpoint is forbidden. Suspended quotes are ineligible; the first executable quote after resume must be defined before X-05 label generation.

## 4. Experiment-specific gates

- **X-01 / X-09:** Replay Levels 1 and 2 are mandatory. Equal P&L with a different event order or canonical hash fails. Level 3 byte identity is not required.
- **X-02 / X-03 / X-07:** These are measurement exemptions. A falsified directional hypothesis may still produce a valid completed measurement; it does not become a positive promotion decision.
- **X-04:** Report each ±1/2/5/10/30/60 second tolerance. “Minute-scale researchable, second-scale indeterminate” is a valid conclusion. No unregistered effect threshold is invented.
- **X-06:** Report Brier, log loss, calibration slope and intercept, and game-cluster bootstrap confidence intervals. Gate 1 (Reaction Model) and Gate 2 (trading relevance/cost comparison) are separate; Gate 1 never implies Gate 2.
- **X-08:** Seven days means wall-clock elapsed capture, not fixture duration. “No gaps” is the pass criterion; uptime below 99% triggers repair and rerun.
- **X-10:** 45/50 passes the research precision gate; 44/50 does not. A pass never authorizes live arbitrage.

## 5. Promotion and compliance

Team H can veto promotion. Team I green for the relevant venue, jurisdiction, account type, data license, and operation is required before any real-money action. Until then, every order and fill is simulated or paper-only.

The following remain closed in v0:

1. real-money execution;
2. live maker strategy or trained maker queue point estimate;
3. exact PMXT L2 queue-fill claims;
4. live multi-venue arbitrage;
5. live copy trading;
6. LLM in the hot path;
7. reinforcement learning;
8. large-scale microservice decomposition;
9. unregistered backtests;
10. treating README returns as evidence.

Failure is evidence when it follows the registered design. It is not permission to widen scope.
