# Whale-Activity Module Report v0

- **Owner:** Team G1
- **Version:** v0
- **Due gate:** `2026-08-19_W4_review`
- **Scope:** `NO_DEMO_NO_LIVE`

## Evidence

R-041 is an advanced evidence-card assignment on non-retail fill-side activity. It motivates measurement, not a causal alpha claim.

## Real execution difficulty

Public identity may be incomplete; one actor can use many accounts; fills mix informed trading, inventory transfer, hedging, and market making. Detection is delayed and large prints can exhaust the price being copied.

## Honest backtest design

Freeze entity-clustering rules, classify only prior activity, measure signal publication latency, execute at subsequent ask/bid depth, and use actor- and market-grouped walk-forward splits. Report false linkage and survivorship sensitivity.
