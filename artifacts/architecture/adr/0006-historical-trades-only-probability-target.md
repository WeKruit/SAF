# ADR 0006: Historical Trades-Only Probability Target

- Status: Accepted
- Date: 2026-07-29
- Decision owner: Team A + Team H
- Evidence trigger: X-15 exact-153 historical rule/continuity audit

## Context

The exact-153 NFL development cohort has complete, paginated historical trades
from Polymarket and Kalshi, but it does not have point-in-time Polymarket tick
rules or a historical suspension/continuity stream for either venue. Kalshi's
official 2025 platform documentation supports a one-cent platform minimum, but
that is not a per-contract rule snapshot. Current Polymarket metadata cannot
prove the tick that applied at a 2025 event.

Using a current rule, an inferred trade grid, or absence of trades as proof of
continuous venue availability would violate ADR 0005.

The historical research question is narrower than execution:

> Conditional on a clean NFL information window and an actually observed later
> trade, did the observed home-outcome probability move up, down, or less than a
> pre-registered economically meaningful amount?

## Decision

Keep the original venue-rule-dependent and market-continuity-dependent targets
fail-closed unless their historical evidence packets are available.

Add a separate diagnostic target for the exact-153 trades-only study:

```text
historical_materiality = 0.01 probability = 1 percentage point

DIAGNOSTIC_UP       delta_L_H >= +0.01
DIAGNOSTIC_DOWN     delta_L_H <= -0.01
DIAGNOSTIC_NO_MOVE  abs(delta_L_H) < 0.01
```

The threshold is a cross-venue research materiality threshold. It is not a
venue tick and must never be named, displayed, or consumed as one.

For historical survival/availability:

- `sports_clean_H` means no next finalized NFL information event or game end
  before H;
- `actual_trade_observed_H` means an actual home-outcome trade exists at H under
  the frozen staleness rule;
- market suspension and feed continuity remain `UNKNOWN`, not `false`;
- an observed pre/post trade pair supports a probability-path observation but
  does not prove continuous market availability between the two observations.

All outputs carry:

```text
claim_boundary = HISTORICAL_TRADES_ONLY_SOURCE_TIME_PROBABILITY_DIAGNOSTIC
venue_tick_support = UNSUPPORTED unless a historical rule packet exists
market_continuity_support = UNKNOWN unless a historical continuity packet exists
```

The diagnostic target may be used for factor discovery, chronological
development OOF, and Polymarket-to-Kalshi development transport. It cannot pass
an execution, latency, L2, tick-reaction, t50/t90, or tradability promotion gate.

X-14 prospective capture continues to use actual point-in-time venue rules,
snapshot/delta continuity, and local monotonic receive times.

## Consequences

The 153-game historical probability-direction study can proceed without
inventing rule or continuity evidence. Its result remains a retrospective
source-time correlation. The confirmatory venue-tick target remains blocked
where evidence is absent, and the two target families cannot be pooled.

## Program NO-GO preserved

This decision authorizes no real-money execution, maker model, hedging, order
placement, fill claim, cross-venue live arbitrage, live copy trading, RL, LLM
hot path, or fabricated historical BBO/L2.
