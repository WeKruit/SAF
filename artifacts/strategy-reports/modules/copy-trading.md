# Copy-Trading Module Report v0

- **Owner:** Team G1
- **Version:** v0
- **Due gate:** `backlog`
- **Scope:** `NO_DEMO_NO_LIVE`

## Evidence

I-023 is a background implementation catalog entry only. It is not evidence that copied accounts remain identifiable, timely, or profitable.

## Real execution difficulty

Follower observations arrive after leader action and may omit cancels, inventory, hedges, private fills, wallet linkage, and intent. Copying changes price and receives a worse queue/depth state.

## Honest backtest design

Predefine leader selection out of sample, use only observations available to the follower, apply lag/ask/depth/fee costs, cluster by leader and market, and include delisted or failed leaders. Live copy trading remains NO-GO.
