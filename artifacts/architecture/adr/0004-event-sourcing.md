# ADR 0004: Append-Only Event Sourcing

- Status: Accepted
- Date: 2026-07-22
- Decision owner: Team A

## Context

Point-in-time research, rule changes, conflicting observations, and X-09 deterministic replay cannot be audited if records are updated in place or historical values are backfilled from current state.

## Decision

The system uses **event sourcing** with an **append-only** raw layer and append-only canonical event stream. Raw objects are retained byte-for-byte, addressed by lowercase SHA-256, and paired with capture session and ordinal. Corrections, mapping changes, rule changes, and experiment amendments create new events or assertions; they never overwrite prior observations.

Normalized state must be possible to **reconstruct** from raw objects, deterministic order, the canonical-ID registry snapshot, versioned venue-rule snapshots, configuration, dependency lockfile, and registered seed. Conflicting observed venue facts remain separate snapshots. Runtime telemetry is not inserted into canonical identity.

## Consequences

Storage consumers use as-of joins and explicit ordering. Derived events retain parent event IDs. Deletion, mutation, and use of current rules to rewrite historical simulation are contract violations.

## Program NO-GO preserved

This decision authorizes none of the blocked scope: real-money execution; maker queue strategies or exact queue-fill claims from PMXT L2; multi-venue live arbitrage; live copy trading; an LLM hot path; reinforcement learning; large-scale microservices; F1 or MLB productionization; a self-built AMM; on-chain market making; strategy selection from README return claims; or unregistered fast backtests.
