# ADR 0002: Native Adapter and Unified Control-Plane Boundary

- Status: Accepted
- Date: 2026-07-22
- Decision owner: Team A

## Context

Venue wire semantics determine whether capture, recovery, and order state are correct. A unified API is useful for discovery and research but cannot erase venue-specific authentication, sequence, pause, cancellation, or lifecycle rules.

## Decision

Each **native adapter** owns its venue wire protocol, authentication and key use, rate limits, reconnect and resubscribe logic, native sequence handling, byte-exact raw capture, and native order acknowledgements and state. The unified **control plane** owns canonical IDs, normalized event envelopes, deterministic replay, simulation, experiment validation, and slow-path discovery and metadata. The boundary is the append-only raw record and the versioned canonical contracts; venue SDK objects never become canonical records directly.

PMXT may serve discovery, metadata, archive research, and Router evidence in the control plane. It is not the hot-path transport and its L2 data does not establish exact queue position. Paper and live OMS components consume the same versioned rule-snapshot stream.

## Consequences

Adding a venue requires a native adapter and evidence-backed mapping assertions, not conditionals in a global venue service. Unknown native states cross the boundary as explicit failures or quality events; the control plane does not invent a normalized success state.

## Program NO-GO preserved

This decision authorizes none of the blocked scope: real-money execution; maker queue strategies or exact queue-fill claims from PMXT L2; multi-venue live arbitrage; live copy trading; an LLM hot path; reinforcement learning; large-scale microservices; F1 or MLB productionization; a self-built AMM; on-chain market making; strategy selection from README return claims; or unregistered fast backtests.
