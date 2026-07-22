# ADR 0001: Engine Decision Is Deferred Until X-09 Evidence

- Status: Accepted
- Date: 2026-07-22
- Decision owner: Team A

## Context

The program needs a deterministic vertical slice before it can know whether an external trading engine improves correctness or only adds translation boundaries. The Charter directs Team B to evaluate NautilusTrader against a lightweight modular monolith and explicitly says that the evaluation is not a selection.

## Decision

The engine decision is deferred until X-09 produces Level 1 and Level 2 deterministic replay evidence. **No engine selected** is the approved architecture state for v0. X-09 evidence must cover native adapter integration, canonical event order and lineage, restart recovery, rule-snapshot consumption, and reproducible orders, fills, P&L, and stream hash. Team A records a later selection only as a new approved ADR after that evidence exists; this ADR does not reopen the approved v0 architecture.

## Consequences

The foundation remains a contract-first modular monolith. Components depend on v0 contracts rather than framework-specific types. An SDK is not treated as a complete trading backend. Work needed solely to integrate a candidate engine waits for the evidence decision.

## Program NO-GO preserved

This decision authorizes none of the blocked scope: real-money execution; maker queue strategies or exact queue-fill claims from PMXT L2; multi-venue live arbitrage; live copy trading; an LLM hot path; reinforcement learning; large-scale microservices; F1 or MLB productionization; a self-built AMM; on-chain market making; strategy selection from README return claims; or unregistered fast backtests.
