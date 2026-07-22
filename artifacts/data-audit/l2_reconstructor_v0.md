# PMXT Deterministic Level-2 Reconstructor v0

- **Owner:** Team C
- **Code:** `prediction_market.pmxt.reconstructor`
- **Evidence source:** PMXT v2 original SHA-256 `590216674e38f820fa011362967124603d3aa28959d05fb9b54b6ca4d3bcf399`
- **Scope:** visible Level-2 price levels only

## Contract

The reconstructor applies only two PMXT event types:

1. `book` replaces both sides for one `(market, asset_id)` stream from the official `[price, size]` pairs.
2. `price_change` updates bids for `BUY`, asks for `SELL`, and removes a level when delta size is zero.

Prices and sizes are parsed as `Decimal` and emitted as canonical decimal strings. Binary floats, naïve timestamps, non-millisecond timestamps, malformed book pairs, and invalid sides fail closed.

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

Canonical events use sorted JSON keys, UTF-8, UTC millisecond timestamps, fixed-point decimal strings, and no insignificant whitespace. Exact canonical duplicates are removed by SHA-256. Each Level-2 semantic event contains the source event hash and the complete sorted bid/ask state. The stream hash is SHA-256 over canonical JSON Lines, including one newline per semantic event.

Input enumeration order does not affect semantic events, quality counts, or the stream hash. `OUT_OF_ORDER` means that, after receive-time ordering, source time moved backward within the same `(market, asset_id)` stream. It does not mean the caller supplied files in a particular order.

## Quality semantics

| Flag | Exact meaning |
|---|---|
| `DUPLICATE_EVENT` | Repeated canonical source event |
| `OUT_OF_ORDER` | Source timestamp regressed within a receive-ordered stream |
| `RECEIVE_GAP_CANDIDATE` | Same-stream receive interval exceeded the configured threshold; not confirmed loss |
| `MISSING_INITIAL_SNAPSHOT` | Delta arrived before that asset's first observed snapshot |
| `NONPOSITIVE_SIZE` | Snapshot level had size ≤ 0, or delta had size < 0; zero delta remains the defined delete operation |
| `CROSSED_BOOK` | After an applied semantic event, best bid was greater than or equal to best ask |

Counts are event/level observations, not distinct incidents. A time gap is never promoted to confirmed packet loss because no sequence is available.

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

Two independent executions over the same preserved original produced the same stream hash:

```text
sha256:3fadfe6daa685b92b8cb2578964dbf405e92b7a1ab6367b8144155adf18fb981
```

The tracked deterministic fixture additionally covers exact deduplication, input-order independence, fixed-point updates, delete semantics, crossed books, nonpositive sizes, missing snapshots, and receive-gap candidates. Its current test-vector stream hash is:

```text
sha256:c0e89df02ed173ab0c8eb704deefdc89c4a5fdd83724a4474e3941707a9ede86
```

## Evidence boundary

This is an engineering reconstruction measurement for one real hour. It is not an X-01 result: the full selected day, an independent implementation comparison, preregistered hashes, and all X-01 locks remain outstanding. Level-1 replay is derivable from these Level-2 states, but no execution or fill conclusion follows from that fact.
