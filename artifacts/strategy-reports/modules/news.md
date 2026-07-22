# News Module Report v0

- **Owner:** Team G2
- **Version:** v0
- **Due gate:** `backlog`
- **Scope:** `NO_DEMO_NO_LIVE`

## Evidence

The current catalog does not provide a dedicated core news-trading artifact. This is an explicit evidence gap; no README or anecdotal return fills it.

## Real execution difficulty

Publication time differs from ingestion time, corrections and duplicates are common, entity-to-market linkage is semantic, and the market may move before a public feed is observable.

## Honest backtest design

Archive original bytes and receive timestamps, deduplicate without future knowledge, preregister entity linkage, use executable quotes after ingestion plus latency, and compare against market prior. The pipeline is slow-path research only.
