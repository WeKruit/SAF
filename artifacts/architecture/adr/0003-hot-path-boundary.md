# ADR 0003: Hot-Path Boundary

- Status: Accepted
- Date: 2026-07-22
- Decision owner: Team A

## Context

Capture and OMS correctness depend on deterministic, bounded work. Discovery, semantic matching, and language-model inference have different latency, availability, and reproducibility properties.

## Decision

The **hot path** consists only of native market-data capture, deterministic normalization, state update, pre-registered signal evaluation, risk checks, simulated or later-approved order state, and immutable logging. It accepts only versioned canonical inputs and cannot call an external discovery service or generative model.

PMXT discovery, archive queries, Router matching, research enrichment, news interpretation, and every LLM operation are the **slow path**. Slow-path output enters canonical processing only through a content-addressed, timestamped event with lineage and, when derived, a registered experiment ID. An LLM cannot place, amend, cancel, or authorize an order.

## Consequences

A slow-path outage cannot stall raw capture or OMS recovery. The v0 design does not promise low latency from PMXT or LLM services and does not add fallback semantics that could silently change trading logic.

## Program NO-GO preserved

This decision authorizes none of the blocked scope: real-money execution; maker queue strategies or exact queue-fill claims from PMXT L2; multi-venue live arbitrage; live copy trading; an LLM hot path; reinforcement learning; large-scale microservices; F1 or MLB productionization; a self-built AMM; on-chain market making; strategy selection from README return claims; or unregistered fast backtests.
