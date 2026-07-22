# Arbitrage Module Report v0

- **Owner:** Team G1
- **Version:** v0
- **Due gate:** `2026-08-05_W2_review`
- **Scope:** `NO_DEMO_NO_LIVE`

## Evidence

R-023 is the core NBA-market evidence card; R-025 adds capital lock-up as an advanced constraint. X-10 supplies semantic precision evidence before any arbitrage study may advance.

## Real execution difficulty

Displayed spreads omit taker fees, delay, depth, partial fills, leg risk, capital fragmentation, suspension, and asymmetric resolution. The Charter permanently blocks simultaneous live multi-venue execution in this phase.

## Honest backtest design

Use point-in-time books and rule snapshots, executable depth on both legs, asynchronous latency scenarios, unresolved-leg bounds, and capital-time cost. Treat unmatched resolution terms as incompatible rather than profitable.
