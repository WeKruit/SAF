# ADR 0005: Fail Closed on Unknown Orders, Rules, and Gaps

- Status: Accepted
- Date: 2026-07-22
- Decision owner: Team A

## Context

An unknown order state can duplicate exposure; an unknown rule can fabricate execution economics; an unexplained source gap can make replay appear deterministic while omitting the market transition that mattered. Defaults would convert absence of evidence into false facts.

## Decision

The program will **fail closed** for an **unknown order** state, **unknown rule** value, or unexplained data **gap**. OMS action stops for the affected order or market until reconciliation establishes native truth. Simulation and replay reject a missing per-condition rule snapshot rather than substituting zero fees, zero delay, a global venue constant, or today's rules. Reconstruction marks and reports gaps; a gap without an accepted classification blocks dependent labels, models, and backtests.

Missing or ambiguous execution values are validation errors. Conflicting observed values are retained as independent, dated snapshots and selected only by point-in-time as-of semantics. No retry path may reinterpret an unknown as accepted, cancelled, filled, or safe.

## Consequences

Availability may be lower during uncertainty, but evidence remains honest and order exposure remains bounded. Recovery requires an explicit reconciliation or new observed snapshot; no compatibility, fallback, or silent degradation path changes business semantics.

## Program NO-GO preserved

This decision authorizes none of the blocked scope: real-money execution; maker queue strategies or exact queue-fill claims from PMXT L2; multi-venue live arbitrage; live copy trading; an LLM hot path; reinforcement learning; large-scale microservices; F1 or MLB productionization; a self-built AMM; on-chain market making; strategy selection from README return claims; or unregistered fast backtests.
