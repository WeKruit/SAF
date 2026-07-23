# PMXT Deterministic Level-2 Reconstructor v0

- **Owner:** Team C
- **Code:** `prediction_market.pmxt.reconstructor`
- **Evidence source:** PMXT v2 original SHA-256 `590216674e38f820fa011362967124603d3aa28959d05fb9b54b6ca4d3bcf399`
- **Scope:** visible Level-2 price levels only

## Contract

The reconstructor applies only two PMXT event types:

1. `book` replaces both sides for one `(market, asset_id)` stream from the official `[price, size]` pairs.
2. `price_change` updates bids for `BUY`, asks for `SELL`, and removes a level when delta size is zero.

Prices and sizes are parsed as `Decimal` and emitted as canonical `FixedPointV0 {atoms, scale}` values. Every semantic state is a validated `EventEnvelopeV0` with event type `normalized_observation`, `X-01` experiment lineage, native Polymarket market/asset references, and a parent event ID deterministically derived through a validated source-row envelope. Canonical IDs remain null until an ID-registry snapshot resolves them; the reconstructor does not fabricate mappings. Binary floats, naïve timestamps, non-millisecond timestamps, malformed book pairs, and invalid sides fail closed.

The implementation does **not** reconstruct queue position, order identity, trade matching, or fills. `queue_fill_reconstructed` is always `false`. PMXT has no exchange sequence field, so none of those claims can be recovered from this stream.

## Deterministic replay and hash

Parquet input is first merged with an explicit database `ORDER BY` on:

```text
timestamp_received, timestamp, market, asset_id
```

The final in-memory ordering adds canonical event SHA-256 as the fifth key, yielding the Charter key:

```text
receive_ts → source_ts → market → asset_id → payload_hash
```

Canonical source rows use sorted JSON keys, UTF-8, UTC millisecond timestamps, fixed-point decimals, and no insignificant whitespace. Exact canonical duplicates are removed by SHA-256. Each semantic envelope contains the source event ID/hash and complete sorted bid/ask state.

The Level-2 hash is produced only by the repository's accepted `level2_stream_sha256` implementation. It validates and internally orders every `EventEnvelopeV0`, then hashes the `prediction-market:event-stream:v0\x00` domain tag, one `uint64be` event count, and each ordered event's fixed 32-byte `event_id` digest. It is not a JSONL or delimiter-based hash.

Input enumeration order does not affect semantic events, quality counts, or the stream hash. `out_of_order` means that, after receive-time ordering, source time moved backward within the same `(market, asset_id)` stream. It does not mean the caller supplied files in a particular order.

## Quality semantics

| Flag | Exact meaning |
|---|---|
| `duplicate_event` | Repeated canonical source event |
| `out_of_order` | Source timestamp regressed within a receive-ordered stream |
| `missing_initial_snapshot` | Delta arrived before that asset's first observed snapshot |
| `non_positive_size` | Snapshot level had size ≤ 0, or delta had size < 0; zero delta remains the defined delete operation |
| `crossed_book` | After an applied semantic event, best bid was greater than or equal to best ask |

These flags are validated against the closed `contracts/quality-flags/v0.yaml` vocabulary. Counts are event/level observations, not distinct incidents. `gap_candidates` remains a diagnostic counter only: a receive-time interval is never promoted to the canonical `gap_detected` flag because PMXT provides no sequence evidence.

## Real NBA-hour measurement

The selected condition is the `Thunder vs. Spurs` moneyline:

```text
0x3904885b86a86a2e20b76d92e3eb91ba5e9d6cd3ed294ee759c5f638fe8ec9bb
```

The pushed-down query over `polymarket_orderbook_2026-05-28T23.parquet` observed:

| Counter | Value |
|---|---:|
| Input / unique events | 9,980 / 9,980 |
| `book` events | 1,291 |
| `price_change` events | 8,051 |
| Level-2 semantic events | 9,342 |
| Exact duplicates | 0 |
| Source-time regressions | 8 |
| Receive-gap candidates at 60 s | 0 |
| Deltas before initial snapshot | 14 |
| Nonpositive-size anomalies | 0 |
| Crossed-book observations | 0 |

Forward and reverse input enumeration over the same preserved original produced 9,342 validated envelopes and the same contract-framed stream hash:

```text
sha256:a812eee2881b1a745da6e484c5a4c0be27233c10a4b445fd22cfecccb49103b7
```

The tracked deterministic fixture additionally covers exact deduplication, input-order independence, fixed-point updates, delete semantics, crossed books, nonpositive sizes, missing snapshots, and receive-gap candidates. Its current test-vector stream hash is:

```text
sha256:6f2874def23b05d9d450f1d7858a51c8e2c9d1a8a8c6fea89a92ecb9a8289d4e
```

## Evidence boundary

This is an engineering reconstruction measurement for one real hour. It does not complete Charter Phase 0 or Phase 1 and is not an X-01 result: the full selected day, an independent implementation comparison, preregistered hashes, and all X-01 locks remain outstanding. Level-1 replay is derivable from these Level-2 states, but no execution or fill conclusion follows from that fact.
