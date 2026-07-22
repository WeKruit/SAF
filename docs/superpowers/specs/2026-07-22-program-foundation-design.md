# Prediction Market Research Program Foundation Design

**Status:** Approved for implementation on 2026-07-22  
**Program state:** `CONDITIONAL_GO`

## Authority

The only program-level sources of truth are:

1. `charter/research_program_charter_v0.2.md`
2. `charter/catalog_registry.csv`
3. `charter/catalog_team_assignments.csv`

All other documents, reports, source catalogs, API documentation, and datasets are evidence inputs. They may not override the three program sources of truth. A SHA-256 source manifest records their immutable baseline.

## Scope

This implementation establishes the first program execution layer for Teams A through I:

- versioned cross-team contracts and ADRs;
- experiment and artifact registries;
- immutable raw capture and deterministic replay primitives;
- Polymarket public recording and Kalshi authentication/recorder framework;
- PMXT Phase 0/1 audit and deterministic L2 reconstruction;
- NBA, NFL, and soccer baseline POCs with calibration tooling;
- barrier-label, taker-simulation, and TCA contracts;
- matched-cluster and popular-module evidence workflows;
- compliance and data-license registers;
- the X-09 deterministic vertical slice.

The implementation does not execute real-money orders, promote maker queue models, claim exact queue fills from PMXT L2, execute multi-venue arbitrage, perform live copy trading, place LLMs in the hot path, use RL, or create a microservice architecture.

## Architecture

The repository is a contract-first modular monolith. Native venue adapters own wire protocols, authentication, rate limits, reconnect behavior, native sequence handling, and exact raw capture. The unified control plane owns canonical IDs, normalized envelopes, deterministic replay, simulation, validation, and slow-path discovery/metadata.

Large raw objects are not stored in Git. Git stores schemas, code, immutable manifest references, experiment registrations, reports, and tests. Raw data is append-only and content-addressed; normalized data must be reproducible from raw inputs, the canonical-ID snapshot, venue-rule stream, configuration, and dependency lockfile.

## Determinism

- All canonical timestamps are UTC.
- Price uses integer ticks or fixed-point decimal; size, rates, fees, and probabilities use fixed-point decimal.
- Binary floating-point values are forbidden in canonical serialized contracts.
- Canonical event order follows receive time, source time, market, asset/outcome, and payload hash.
- Level 1 semantic determinism and Level 2 canonical-stream SHA-256 are MVP gates.
- Runtime wall-clock processing times are operational metadata and are excluded from canonical hashes.

## Registries

Team H owns the experiment registry. An experiment result is valid only when its experiment card was registered and content-addressed before evaluation results were inspected. Amendments are append-only and preserve the prior version.

The existing `R-`, `I-`, and `O-` catalog IDs are governance lineage only. Domain canonical IDs use separate opaque namespaces for competition, participant, game, venue event, market, outcome, and condition.

## Raw Capture

WebSocket frames are stored byte-for-byte with local receive time, capture session, record ordinal, and payload SHA-256. Sealed segments receive an immutable sidecar manifest and exact-object SHA-256. PMXT files are retained as downloaded Parquet bytes. Staging files are never visible to consumers.

## Experiment Boundaries

- X-01 through X-10 are registered before execution results are accepted.
- X-07 stays `PRELIMINARY` until versioned venue rules are consumed by replay.
- X-09 failure freezes strategy backtests.
- X-08 needs seven actual elapsed days and cannot be satisfied by synthetic time.
- Kalshi live capture remains blocked until an API key and RSA private key are securely provisioned.
- X-05 label generation remains blocked until Team H locks U/L/h and purge/embargo values.
- X-10 recall is not reported until a valid candidate-universe denominator is registered.

## Verification

The program foundation is verified by:

1. source-of-truth hash and catalog-junction invariants;
2. schema and forbidden-float tests;
3. raw-segment immutability and hash tests;
4. recorder protocol and reconnect tests using local fake transports;
5. deterministic L2 reconstruction and anomaly-flag tests;
6. grouped chronological split and calibration-metric tests;
7. executable ask-entry/bid-exit label tests;
8. fee/delay/depth-aware taker simulation tests;
9. two identical X-09 replays producing identical Level 1 results and Level 2 hash;
10. a final full test, registry-validation, and NO-GO audit.

