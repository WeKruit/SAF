# X-13 Research Dashboard Design

Date: 2026-07-25  
Status: Approved  
Scope: `PRELIMINARY_SOURCE_TIME_ONLY`

## Purpose

The dashboard exists to help a human researcher:

1. replay and inspect one NFL game;
2. query events, game state, contracts, and source-time market observations;
3. visually identify candidate relationships between game events and market
   paths for later registered analysis.

It does not generate causal, latency, execution, or tradable-alpha claims.

## Source-of-truth boundary

The immutable X-13 batch, raw manifests, checksums, and full Parquet tables
remain the research source of truth. The database and dashboard assets are
rebuildable indexes and presentation projections only.

Every published dashboard asset must bind:

- experiment ID;
- batch ID;
- source artifact SHA-256;
- projection schema version;
- projection object SHA-256;
- builder version.

Changing the database schema or interface must not mutate an existing batch or
overwrite a content-addressed asset. A changed projection produces a new
versioned object and a new publish manifest.

## Storage design

Use the existing private Supabase S3 bucket under an independent prefix:

```text
prediction-market/dashboard/x13/
  batch=<batch-id>/
    catalog/<sha256>.json
    games/<game-id>/core/<sha256>.json
    games/<game-id>/contracts/<sha256>.json
    games/<game-id>/markets/<venue>/<family>/<market-id>/<sha256>.json
    games/<game-id>/associations/<sha256>.json
    manifests/<sha256>.json
  published/latest.json
```

The PMXT prefix and PMXT runtime hydration module are outside this dashboard's
interface. Full canonical game JSON and association Parquet remain private and
are not copied into a public presentation bucket.

The first database schema contains only:

- `dashboard_batches`: immutable batch identity, status, manifest, publication
  time, and claim boundary;
- `dashboard_games`: matchup, kickoff, final score, coverage counts, and audit
  status;
- `dashboard_assets`: immutable object key, media type, byte length, SHA-256,
  schema version, and logical role.

The database is not used for the 48,624,912 association rows. It can be
migrated or rebuilt from the publish manifest. Contract- or episode-level
tables are deferred until a demonstrated cross-game query needs them.

## Projection assets

The exporter must verify the published X-13 batch before reading it. It emits
bounded, independently loadable assets:

- `catalog`: 20-game list, coverage, publication status, and asset links;
- `core`: teams, score, events, finalized episodes, personnel coverage, and
  cumulative stat ledger;
- `contracts`: normalized propositions, outcomes, coverage, and rule metadata;
- `market`: one logical market's bounded trade ranges and available Kalshi
  candle BBO;
- `associations`: the existing deterministic presentation sample with delay,
  horizon, ambiguity, contamination, and validity fields;
- `publish manifest`: every object path, source hash, projected hash, byte
  length, and schema.

The exporter must not recompute the X-13 experiment, alter association rows, or
invent missing bid/ask data. Derived complements remain visually and
semantically distinct from observed prices.

## Dashboard workflow

The first viewport is the 20-game research index, not generic administration.
It shows matchup, score, play/episode/contract counts, venue coverage, and audit
gate.

Selecting a game opens one research workspace:

1. an American-football field and play scrubber;
2. a source-time event timeline with event/state filters;
3. game-state and personnel details for the selected play;
4. venue, family, and logical-market selectors;
5. two-outcome market paths, observed/derived distinction, and available BBO;
6. event-association filters for delay, horizon, ambiguity, contamination, and
   validity;
7. lineage, hashes, coverage gaps, and claim boundary.

The client loads the catalog first, then game core, and only fetches contract,
market, or association assets when the researcher opens those views. No page
loads the current 399 MB single-game JSON or 42 MB standalone HTML.

## Access and security

Version one is private-first. S3 credentials never enter browser code. The
runtime issues short-lived signed reads or serves verified projection objects.
The existing `.env.s3` is used only by local/server-side publication tooling.

A public read-only presentation bucket is a later deployment option only after
Team I approves redistribution of the selected derived fields. PMXT, raw
captures, full canonical game JSON, and full association Parquet are never made
public by that decision.

## Error handling and audit

The exporter and runtime fail closed when:

- batch verification fails;
- an expected source or projected hash differs;
- an asset is missing from the publish manifest;
- a database row points to a different batch or object hash;
- an unknown schema version is requested;
- a latest pointer references an unpublished manifest.

The interface displays unavailable data explicitly and does not forward-fill
market observations across empty windows.

## Verification

Automated checks cover:

- deterministic projection hashes across two exports;
- byte tampering and manifest tampering;
- exact 20-game catalog coverage;
- lazy asset boundaries and maximum payload sizes;
- two outcomes per logical market where defined;
- observed versus derived rendering semantics;
- absence of fabricated Polymarket BBO;
- database migration and index rebuild from the publish manifest;
- no S3 secrets in browser bundles;
- no unregistered external network requests.

Browser acceptance covers the index, a high-volume game, an overtime game, a
tie, and a no-touchdown game.

## Completion boundary

Version one is complete when a researcher can use the dashboard to move from a
game event to the corresponding source-time market window, inspect the data and
audit status, and copy stable batch/game/episode/market identifiers for a
subsequent registered exploration.

It does not run the unresolved X-13 estimand, model fair value, bootstrap,
trading strategy, or execution simulation.
