# PMXT Phase 1 Schema and Timestamp Audit v0

- **Owner:** Team C
- **Observed:** 2026-07-22
- **Primary evidence:** `phase1_sample_measurement.json`, `phase1_nba_hour_sample.json`
- **Status:** schema/timestamp audit complete for two real hourly originals; Charter one-market/full-day validation remains open

## Immutable source samples

| Purpose | PMXT object | Exact bytes | Original SHA-256 | Rows | Row groups |
|---|---|---:|---|---:|---:|
| Bounded schema sample | `polymarket_orderbook_2026-05-22T18.parquet` | 1,611,563 | `sha256:f7e71296c6163760c94389a40458ec3d9e9769f92975c5f68c1c44b313ff7661` | 224,332 | 1 |
| NBA-hour sample | `polymarket_orderbook_2026-05-28T23.parquet` | 436,334,049 | `sha256:590216674e38f820fa011362967124603d3aa28959d05fb9b54b6ca4d3bcf399` | 75,587,694 | 73 |

Both files were downloaded from `https://r2v2.pmxt.dev/<filename>`, preserved unchanged under a SHA-256 content address, and audited from the preserved object. The raw objects remain under the gitignored `var/raw/pmxt-v2/sha256/` store; the committed reports pin URL, exact byte length, and hash. Publication uses a same-directory fully synced staging file followed by atomic no-overwrite publication. Every path component is opened without following symlinks, published objects are read-only, and every reuse or archived read verifies bytes against the filename content address.

The bounded sample was selected as the smallest currently listed object, breaking equal-size ties toward the later hour. It is useful for schema verification but is not representative of market activity. The second object was deliberately selected because official Gamma metadata identifies condition `0x3904885b86a86a2e20b76d92e3eb91ba5e9d6cd3ed294ee759c5f638fe8ec9bb` as the `Thunder vs. Spurs` NBA moneyline market (`nba-okc-sas-2026-05-28`). The metadata lookup was <https://gamma-api.polymarket.com/markets?condition_ids=0x3904885b86a86a2e20b76d92e3eb91ba5e9d6cd3ed294ee759c5f638fe8ec9bb>.

## Observed v2 schema

Both originals have the same 16-column schema and no schema deviations:

| Column | Arrow type | Declared nullable |
|---|---|---|
| `timestamp_received` | `timestamp[ms, tz=UTC]` | no |
| `timestamp` | `timestamp[ms, tz=UTC]` | no |
| `market` | `fixed_size_binary[66]` | no |
| `event_type` | `string` | no |
| `asset_id` | `string` | no |
| `bids`, `asks` | `string` | yes |
| `price`, `best_bid`, `best_ask`, `old_tick_size`, `new_tick_size` | `decimal128(9, 4)` | yes |
| `size` | `decimal128(18, 6)` | yes |
| `side`, `transaction_hash` | `string` | yes |
| `fee_rate_bps` | `uint16` | yes |

`bids` and `asks` are JSON strings containing PMXT pairs of `[price, size]`. The reconstructor parses those values into exact decimals; binary float is rejected.

## Event and timestamp observations

| Measure | Bounded sample | NBA-hour sample |
|---|---:|---:|
| `book` | 556 | 206,571 |
| `price_change` | 222,282 | 75,313,012 |
| `last_trade_price` | 1,482 | 67,815 |
| `tick_size_change` | 12 | 296 |
| `timestamp_received - timestamp` minimum | 21 ms | 20 ms |
| median | 301 ms | 219 ms |
| p95 | 506 ms | 5,064 ms |
| maximum | 176,216 ms | 2,552,328,918 ms |
| negative deltas | 0 | 0 |

The NBA-hour receive range is 2026-05-28T23:00:00.103Z through 23:59:59.961Z, while its oldest source timestamp is 2026-04-29T08:58:39.323Z. The 29-day maximum delta is direct evidence that source time cannot be used as the primary replay clock. Global replay must start with `timestamp_received` and retain `timestamp` only as the second tie-break/audit dimension.

Null counts for every column are in the machine reports. Timestamp, market, event type, and asset ID were non-null for every row in both samples.

## Remaining gate

The NBA evidence covers one market in one hourly object, not the complete UTC day. Its optional reconstruction summary records 9,980 inputs, 9,342 validated `EventEnvelopeV0` states, and contract-framed Level-2 hash `sha256:a812eee2881b1a745da6e484c5a4c0be27233c10a4b445fd22cfecccb49103b7`. It therefore validates the implementation and sample boundary but does not complete Charter Phase 0/1, X-01, X-02, or X-03. A formal one-market/day result still requires all selected-day objects, a frozen market-metadata snapshot, preregistered hashes, and the independent reconstruction comparison required by X-01.

## Reproduction

```bash
uv run python -m prediction_market.cli.audit_pmxt sample \
  --max-files 1 \
  --raw-root var/raw \
  --output artifacts/data-audit/phase1_sample_measurement.json

uv run python -m prediction_market.cli.audit_pmxt sample \
  --url https://r2v2.pmxt.dev/polymarket_orderbook_2026-05-28T23.parquet \
  --max-files 1 \
  --max-bytes 600000000 \
  --condition-id 0x3904885b86a86a2e20b76d92e3eb91ba5e9d6cd3ed294ee759c5f638fe8ec9bb \
  --raw-root var/raw \
  --output artifacts/data-audit/phase1_nba_hour_sample.json
```
