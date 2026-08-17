# Trading Layer Design v0 — Kalshi + Polymarket Execution Architecture

- Status: **PROPOSAL** (not an accepted ADR; authorizes nothing)
- Date: 2026-08-17
- Owner: A+B+C (architecture + venue adapters + execution)
- Registry: `ART-A-021`

**Governance boundary.** This document proposes the execution-plane architecture
for the program's two venues. It does not amend the charter, does not select an
engine (ADR-0001 keeps that decision deferred until X-09-class evidence), and
does not unlock any blocked scope. Real-money execution, live maker promotion,
multi-venue simultaneous live arbitrage, copy trading, LLM hot path, RL, and
large-scale microservices remain **NO-GO** per
`charter/research_program_charter_v0.2.md` §0.5. Every phase below that touches
an exchange runs against demo environments or in shadow mode until the
program's promotion gates pass and the charter is amended.

**Evidence discipline.** Venue facts below were verified against official
documentation on 2026-08-16/17 (sources cited inline). Facts that could not be
confirmed first-party are marked `[VERIFY]` and appear again in §9's
verification checklist. Per ADR-0004/0005 discipline, none of the numbers here
(fees, delays, limits) may be hard-coded into simulation or execution: they
enter the system only through versioned `VenueRuleSnapshotV0`-class records
captured from the venue at runtime.

---

## 1. Executive summary

Both venues are cloud-hosted services, not exchange-colocated matching engines
with multicast feeds. That single fact drives the whole design:

1. **The latency floor is network placement, not code.** Kalshi's engine is
   reachable at ~10 ms RTT from the right region (community evidence puts it in
   AWS us-east-2 `[VERIFY]`, with AWS PrivateLink officially offered at Premier
   tier — the closest thing to colocation). Polymarket's CLOB origin is
   community-triangulated to AWS eu-west-2 behind Cloudflare `[VERIFY]`, with
   ~13–15 ms WebSocket delivery and ~21–23 ms warm order round-trips measured
   from Dublin. Sub-millisecond in-process engineering is table stakes we get
   almost for free; the real wins are region placement, warm connections,
   session-level auth, and rate-tier progression.

2. **"FIX" is real but narrower than it sounds.** Kalshi offers a full FIX
   surface (FIXT.1.1 session layer, FIX 5.0 SP2 application layer: order entry,
   market data, drop copy, post-trade, RFQ). Its value is *transport and
   semantics* — one RSA signature at logon instead of ~1 ms of RSA-PSS per REST
   request, persistent TLS session, cancel-on-disconnect, mass-cancel — **not
   rate limits**, because FIX drains the same token buckets as REST. Access is
   default at Premier tier (earned or granted; demo FIX endpoints exist).
   Polymarket's global CLOB has **no FIX at all** — orders are EIP-712-signed
   payloads over REST. FIX exists only at Polymarket US (QCX), a separate
   CFTC-regulated exchange with a separate API stack.

3. **Venue rules changed hard in 2026 and are still changing this week.**
   Polymarket cut over to CLOB V2 on 2026-04-28 (new signed-order struct, new
   contracts, pUSD collateral, no backward compatibility) and removed its
   500 ms taker delay on 2026-02-18. Kalshi deprecated integer-cent prices for
   fixed-point dollars with per-market price bands, introduced 7 volume-earned
   rate tiers, and is **sharding its exchange by category between 2026-08-06
   and 2026-08-24** (collateral must be pre-allocated per shard; explicit
   `exchange_index` routing avoids an auto-route latency penalty). The design
   therefore treats venue rules as streamed data, never constants.

4. **Proposed shape** (consistent with ADR-0002/0003): extend the existing
   governed monolith with an execution plane — two *native execution adapters*
   (Kalshi: WS market data + FIX-first order entry with REST V2 fallback;
   Polymarket: CLOB V2 REST order path + user WebSocket fills + heartbeat
   dead-man switch), one *deterministic OMS core* (single-writer order state
   machines, fail-closed reconciliation per ADR-0005), an in-line *risk gate*
   backed by venue-native protections (Kalshi order groups + cancel-on-
   disconnect; Polymarket heartbeats), and the existing append-only journal and
   TCA contracts. Python-first hot path with measured budgets and an explicit
   Rust escape hatch; the engine decision itself stays deferred per ADR-0001.

5. **Adopt / build split from the OSS survey:** adopt hftbacktest as the
   latency- and queue-aware backtesting spine, Polymarket's official V2 signing
   clients, and a standard FIX engine or a minimal in-house FIXT initiator for
   Kalshi; mine NautilusTrader's Polymarket adapter docs, TexasCoding's
   pure-Python Kalshi FIX engine, and community arb bots for reference. Nothing
   open-source ships a production Kalshi FIX gateway, Kalshi historical L2
   data, or a settlement-aware cross-venue equivalence registry — those are the
   build list, and recording our own L2 from day one is the compounding moat.

---

## 2. What the venues actually offer (2026-08)

### 2.1 Capability matrix

| Capability | Kalshi (Predictions API) | Polymarket (global CLOB V2) |
|---|---|---|
| Regulatory form | CFTC DCM+DCO; direct membership | Off-chain operator CLOB, on-chain non-custodial settlement (Polygon) |
| REST base | `https://external-api.kalshi.com/trade-api/v2` (dedicated external-trader host; `api.elections.kalshi.com` also serves all markets) | `https://clob.polymarket.com` (`GET /version` must return `2`) |
| Demo/paper env | Full parallel stack: `external-api.demo.kalshi.co` (+ WS, FIX) | **None** (no public testnet CLOB) |
| Order transport | REST V2, **FIX FIXT.1.1/FIX50SP2**, WS (data only) | REST only (EIP-712-signed order payloads); WS for data/fills |
| Market data | WS (`orderbook_delta` seq'd snapshot+delta), FIX MD (port 8233, since 2026-05-28), REST snapshots, candlesticks | WS market channel (`book`, `price_change`, `tick_size_change`), RTDS firehose, REST books/prices |
| Auth | API key ID + RSA-PSS(SHA-256) signature per REST request / WS handshake / FIX logon | L1 wallet EIP-712 (mint API creds) → L2 HMAC-SHA256 per request; orders themselves EIP-712-signed |
| Order types | limit (+ market on REST only), GTC/GTD/IOC/FOK/Day(FIX), post-only, reduce-only, STP mandatory on V2 | GTC/GTD/FOK/FAK, post-only, batch ≤15 orders / ≤1,000 cancels |
| Price grid | Fixed-point dollars; per-market `price_ranges` bands, steps $0.01 → $0.0001 (integer-cent fields deprecated 2026-03-05) | Per-token tick from {0.1 … 0.0001}, changes dynamically past 0.96/0.04 (`tick_size_change`) |
| Rate limits | Token buckets (Read/Write) shared REST+FIX; Basic 100W/s → self-serve Advanced 300W/s (≈30 orders/s) → … → Prestige 8,000W/s; create=10 tokens, cancel=2 | Per-signer 40 orders/s (burst 60), 80 cancels/s (burst 120) standard → ~600/s top tier via 30-day maker volume; Cloudflare IP caps (order POST ~5,000/10 s) `[VERIFY: moved twice in 2026]` |
| Fees | Taker `ceil(0.07·C·P·(1−P))` per general table `[VERIFY vs fee PDF]`; maker fees on designated series (~25% of taker); centicent rounding accumulator; per-series `fee_type`/`fee_multiplier` from API | Taker `C·rate·p·(1−p)`, category rates ~0.03–0.07 (geopolitics free) `[VERIFY via GET /fee-rate]`; makers 0 + daily pUSD rebates; fees collected on-chain at match |
| Risk primitives | **Order groups** (auto-mass-cancel when fills in a rolling 15 s window exceed a limit), `CancelOrdersOnDisconnect` (FIX logon flag), `cancel_order_on_pause`, queue-position endpoints | **Heartbeat API** (dead-man: miss ~10 s beat → all orders cancelled), cancel-all endpoint, on-chain fallback cancel if API unreachable |
| Fills/acks fast path | FIX ExecutionReports; WS `fill` channel; own `client_order_id` echoed in `orderbook_delta` | User WS channel (`PLACEMENT`/`MATCHED`/…); trade lifecycle `MATCHED→MINED→CONFIRMED` (on-chain tail) |
| Settlement | Cleared at $1.00/contract by Kalshi (DCO) | Operator submits `matchOrders` on Polygon; trader pays no gas; position final at on-chain confirmation |
| Sharding/scale | **Exchange sharding by category (Aug 2026)**: per-shard collateral pre-allocation, `exchange_index` routing, order groups don't span shards | Single logical book per token; neg-risk multi-outcome via separate exchange contract/domain |
| Proximity product | **AWS PrivateLink** for REST/WS/FIX at Premier+; engine region community-pegged us-east-2 `[VERIFY]` | None official; origin community-pegged AWS eu-west-2 behind Cloudflare `[VERIFY]` |
| Institutional/other | FCM/subaccounts (0–63), block trades, RFQ suite, Perps API (separate) | Builder program (attribution in signed order, relayer), RFQ subsystem, **Polymarket US = separate exchange with Ed25519 REST/WS + FIX 5.0 SP2 gateways (150 msg/s/session) via FCM/ISV onboarding** |

Primary sources: docs.kalshi.com (`getting_started/*`, `fix/*`, `api-reference/*`,
changelog; read 2026-08-16), docs.polymarket.com plus Polymarket's
`py-clob-client-v2` / `ctf-exchange-v2` / `agent-skills` repositories, and
NautilusTrader's Polymarket adapter notes (rate-limit snapshot dated
2026-08-04). Full citation list in §10.

### 2.2 The 2026 change timeline that invalidates older designs

| Date | Change | Design consequence |
|---|---|---|
| 2026-02-18 | Polymarket removes 500 ms taker delay | Maker cancel/replace loop is now genuinely latency-competitive (community target: sub-100–200 ms loop) |
| 2026-03-05 | Kalshi deprecates integer-cent price fields | All price handling in fixed-point dollars + per-market `price_ranges` |
| 2026-03-30 | Polymarket category-wide taker fees ("Fee Structure V2") | Zero-fee assumptions dead; fee rates are per-market runtime data |
| 2026-04-28 | **Polymarket CLOB V2 cutover** (domain v2, struct change, pUSD, new contracts, V1 wiped) | Only V2-era SDKs/signing valid; `GET /version` gate; `timestamp` field replaces nonce `[VERIFY freshness window]` |
| 2026-05-28/29 | Kalshi FIX market data + SecurityStatus streams | FIX-only market data becomes possible at Premier |
| 2026-06-04→25 | Kalshi V2 event-order endpoints; legacy costs 10×; 7-tier volume-earned rate ladder, self-serve Advanced | Build on `/portfolio/events/orders`; plan tier progression |
| 2026-08-06→24 | **Kalshi exchange sharding** (combos → shard 1 on 08-17; crypto/tennis/baseball 08-24) | Shard-aware routing, per-shard collateral management, per-shard `exchange/status` |
| 2026-08-19/21 | Kalshi maker fees enabled (combos); combo quoter fee swap | Fee logic must consult per-series `fee_type` + change-schedule endpoints |

---

## 3. Where the latency actually is

### 3.1 Budget decomposition (event → order at venue)

Magnitudes below are order-of-magnitude engineering estimates except where a
measured source is cited; none are contract inputs.

| Stage | Kalshi (region-adjacent) | Polymarket (region-adjacent) | Notes |
|---|---|---|---|
| Venue event → WS delivery | ~5–15 ms | ~13–15 ms measured (Dublin) | CDN/LB in path; FIX MD may shave Kalshi `[MEASURE]` |
| Parse + book update | 10–100 µs (Python) / 1–10 µs (Rust) | same | Bounded grid → array-indexed book (§5.3) |
| Signal / model inference | 10–100 µs | same | XGBoost via native `inplace_predict`; models pre-loaded, pre-registered (ADR-0003) |
| Risk gate | <10 µs | same | Pure in-memory checks |
| Order construct + sign | **FIX: ~10 µs** (no per-order crypto) / REST: RSA-PSS ≈ 0.5–1.5 ms | EIP-712 secp256k1 ≈ 50–150 µs with C bindings and precomputed type/domain hashes; ms-scale if naïve | This is FIX's main measurable win on Kalshi; on Polymarket, precompute + C-library signing is mandatory |
| Transport one-way | ~5–10 ms (us-east-2-adjacent) `[VERIFY region]` | ~10 ms (eu-west-2-adjacent) `[VERIFY region]`; +≈35–40 ms if crossing the Atlantic | PrivateLink removes public-internet variance for Kalshi at Premier |
| Venue matching + ack | unpublished, ms-scale | unpublished; warm order RTT ≈21–23 ms measured (Dublin) | |
| Fill certainty | ER on FIX / WS `fill` | `MATCHED` on user WS (economic fill) → `MINED/CONFIRMED` on-chain (seconds) | PM position-certainty tail matters for inventory accounting, not signal speed |

Two structural facts sit above all of this:

- **Venue-imposed delays.** The program's 2026-07-22 doc check recorded an
  official 1-second marketable-order delay on sports markets; Polymarket's
  *general* 500 ms taker delay was removed 2026-02-18. Whether an in-game
  sports delay currently applies, per venue and per market, is **the single
  most decision-relevant unverified fact** `[VERIFY §9.1]` — a 1 s venue delay
  makes shaving 5 ms irrelevant for taking on those markets (the game becomes
  model quality and being early into the delay queue), while its absence makes
  the 10–20 ms order path decisive. The existing `VenueRuleSnapshotV0.seconds_delay`
  field and the fail-closed rule store are exactly where this belongs; the
  taker simulator already consumes it.
- **Queue position is a first-class resource.** Kalshi exposes queue-position
  endpoints and echoes our own `client_order_id` in `orderbook_delta`;
  Polymarket rewards makers (rebates, tier upgrades by maker volume) and
  auto-cancel infrastructure exists on both. For maker strategies (blocked
  today, designed-for tomorrow), time priority earned by a fast cancel/replace
  loop is worth more than any further taker-side shaving.

### 3.2 What "low latency" buys, by strategy class

| Game | Binding constraint | Engineering answer |
|---|---|---|
| Taker on game-state change (our NFL models) | Feed latency + model latency (+ any venue sports delay) | Region-adjacent feed ingest, µs-scale inference, pre-armed order templates; venue delay verification decides how much more is worth buying |
| Maker cancel/replace vs adverse selection | Order-path RTT + venue rate budget | FIX session (Kalshi), warm HTTP/2 + presigned-fields pipeline (PM), cancel-first budgeting, order groups/heartbeats as backstops |
| Cross-venue relative value (research only; live arb NO-GO) | Transatlantic RTT ≈70–80 ms round trip is irreducible `[VERIFY regions]` | Per-venue execution PoPs + a slow consistency layer; do not design one box for both venues |

### 3.3 Placement decision

One execution PoP per venue region (a us-east-2-adjacent host for Kalshi, a
eu-west-2-adjacent host for Polymarket `[VERIFY both]`), each running the same
monolith binary configured for its venue; research/control plane stays wherever
it is. This is two deployments of one program, not microservices. Verification
is cheap (§9.1: traceroute FIX hosts — they are NLB-fronted, not CDN-fronted —
and a one-hour cross-region RTT probe) and should precede any placement spend.

---

## 4. Per-venue protocol decisions

### 4.1 Kalshi

**Market data: WebSocket first.** `orderbook_delta` provides seq-numbered
snapshot+delta with gap detection (`seq` contiguity per subscription,
`get_snapshot` resync), 10 s server pings, and our own `client_order_id` echoed
on our book changes — an ack side-channel. Subscribe with `use_yes_price: true`
(Kalshi has announced the default flip). The existing capture adapter already
implements the subscription and parse (`src/prediction_market/adapters/kalshi.py`);
it extends to a live book builder without changing capture semantics. FIX
market data (port 8233) is a Premier-era optimization to be benchmarked against
WS, not a dependency.

**Order entry: FIX-first, REST V2 as the working fallback.**

- FIX facts that matter: FIXT.1.1 transport, FIX50SP2 application layer
  (`DefaultApplVerID=9`); order-entry hosts `mm.fix.elections.kalshi.com:8228`
  (KalshiNR, no retransmission, `ResetSeqNumFlag=Y` mandatory, mass-cancel
  1/s) and `:8230` (KalshiRT, retransmission); TLS 1.2+ mandatory (AWS NLB
  policies); logon carries `RawData<96>` = base64 RSA-PSS signature over
  `SendingTime|MsgType|MsgSeqNum|SenderCompID|TargetCompID` with SendingTime
  within 30 s of server time; **limit orders only** (no market orders over
  FIX); `UseDollars<21005>=Y` for fixed-point prices; STP tag 2964;
  `ExDestination<100>` for explicit shard routing (auto-route `-1` documented
  as adding latency); `CancelOrdersOnDisconnect<8013>=Y`;
  `AlwaysEmitNewBeforeTrade<21026>=Y` to keep the order state machine simple;
  one FIX connection per API key. Demo FIX hosts exist
  (`fix.demo.kalshi.co`). Access: default at Premier; below that, request via
  institutional@kalshi.com — availability of a pre-Premier grant is
  `[VERIFY §9.1]`.
- Why FIX over REST here: eliminates ~0.5–1.5 ms RSA-PSS per order plus HTTP
  framing, gives cancel-on-disconnect and mass-cancel semantics, richer
  ExecutionReport payload (fees, post-trade positions, collateral deltas,
  aggressor flag, shard id in `LastMkt`), and a persistent session immune to
  connection-pool churn. It does **not** raise rate limits (same token
  buckets), so tier progression (Basic → self-serve Advanced at ~30 orders/s →
  volume-earned Expert/Premier) is a parallel workstream regardless of
  transport.
- FIX engine choice (v0 recommendation: **minimal in-house asyncio FIXT
  initiator**): KalshiNR's mandatory `ResetSeqNumFlag=Y` (no resend/gap-fill
  obligations — a gap simply forces re-logon) shrinks the session layer to
  logon/heartbeat/test-request/logout plus tag-value codec for ~8 message
  types, against Kalshi's published dictionary XML. That is a small,
  deterministic, fail-closed surface consistent with this repo's contract
  style, with TexasCoding's pure-Python Kalshi FIX engine and the QuickFIX
  family as references and `quickfix`-bindings/QuickFIX-J/quickfix-rs as the
  fallback if conformance testing exposes gaps. KalshiRT (retransmission),
  listener sessions (read-only ER shadow on a second key), and drop-copy
  replay (3 h ExecID-range lookback) join in the maker phase.
- REST V2 facts that matter until FIX access lands: `POST/DELETE
  /portfolio/events/orders[...]` with fixed-point dollar prices and single-book
  `bid`/`ask` sides; create=10 tokens, cancel=2 (verify at runtime via `GET
  /account/endpoint_costs`); batch endpoints save round-trips, not tokens;
  mandatory `self_trade_prevention_type`; `client_order_id` idempotency;
  `exchange_index` explicit routing; 429s carry no Retry-After.

**Shard awareness (new, mandatory).** Markets/events/series carry
`exchange_index`; collateral is pre-allocated per shard via the intra-account
transfer API; order groups don't span shards; `exchange/status` is per-shard.
The adapter must treat (venue, exchange_index) as the routing key and the
capital allocator must rebalance across shards as categories migrate
(2026-08-17: combos → shard 1; 2026-08-24: crypto, tennis, baseball).

**Venue-native risk rails to adopt:** order groups (auto-mass-cancel when
fills in a rolling 15 s window exceed a limit — precisely the "runaway model"
backstop), `cancel_order_on_pause`, cancel-on-disconnect at the FIX layer, and
the Thursday 03:00–05:00 ET maintenance window (sessions disconnect; sequence
numbers reset; resting orders persist unless flagged) in the scheduler.

### 4.2 Polymarket (global CLOB V2)

**Market data: CLOB WS market channel** (`book` snapshot on subscribe +
`price_change` + `tick_size_change` + `last_trade_price`; optional
`best_bid_ask`/`new_market`/`market_resolved` behind `custom_feature_enabled`),
10 s PING discipline, subscriptions by token ID (cap ~200/connection observed —
shard connections). RTDS (`ws-live-data.polymarket.com`) is a monitoring
firehose, not the execution feed. The existing capture adapter
(`adapters/polymarket.py`) extends unchanged in role.

**Order path: REST V2 with a hot signing pipeline.**

- Auth chain: L1 EIP-712 (`ClobAuthDomain`, version "1", chainId 137) mints L2
  creds (`POST /auth/api-key` / `GET /auth/derive-api-key`); every trading call
  carries the five `POLY_*` headers with HMAC-SHA256 over
  `timestamp+METHOD+path+body`. Note the domain-version trap: auth domain is
  v"1", the *exchange* order domain is v"2".
- V2 signed order struct: `{salt, maker, signer, tokenId, makerAmount,
  takerAmount, side, signatureType, timestamp(ms), metadata, builder}` against
  domain "Polymarket CTF Exchange" version "2" — `taker`, `expiration`,
  `nonce`, `feeRateBps` are gone; `timestamp` is the uniqueness source and its
  server-side freshness window is undocumented `[VERIFY §9.1 before any
  pre-signing design]`. Signature types: EOA=0, POLY_PROXY=1,
  POLY_GNOSIS_SAFE=2, POLY_1271=3 (new deposit wallets). Neg-risk markets sign
  against a different exchange contract — the `neg_risk` flag is per-token
  routing data.
- Hot-path signing budget: precompute domain separators and type hashes once
  per (exchange, chain) pair; keccak + secp256k1 sign via C bindings
  (`coincurve`/libsecp256k1 in Python, `alloy`+`k256`/`secp256k1` in Rust);
  warm the curve context at startup. Target ≤150 µs per order in Python,
  tens of µs in Rust. Everything else about the order (tick-size-snapped
  price, maker/taker amounts in 6-decimal units, funder/signer split for
  proxy wallets) is precomputable per market and cached against
  `tick_size_change`.
- Order types GTC/GTD (60 s vs 3 min minimum-expiry discrepancy `[VERIFY]`),
  FOK/FAK for taking, post-only for quoting; batch ≤15 per POST; cancels up to
  1,000 per DELETE plus `cancel-all` and market-scoped cancel; **heartbeat
  dead-man switch** (`POST /v1/heartbeats`, beat every ~5 s once started, miss
  ~10 s → all orders cancelled) as the standing safety rail; on-chain cancel
  via the exchange contract as the API-unreachable fallback.
- Fills: user WS channel (`PLACEMENT`/`CANCELLATION`/trade `MATCHED`…) is the
  fast economic-fill signal; `MINED/CONFIRMED/RETRYING/FAILED` tracks the
  on-chain settlement tail. Inventory accounting distinguishes
  economically-filled from chain-final; the V2 engine reportedly eliminated
  V1's "ghost fills," but the state machine still models `FAILED` explicitly
  (fail-closed, ADR-0005).
- Capital plumbing (one-time/ops): wrap USDC.e → pUSD via the onramp, approve
  V2 exchanges for pUSD + CTF, per-funder balance/allowance checks via
  `GET /balance-allowance`. The operator pays all match-settlement gas.
- Rate limits are two-layered — per-signer token buckets (≈40 orders/s burst
  60 standard; tiers rise with 30-day *maker* volume) and Cloudflare per-IP
  windows — both are config, refreshed from the live docs/headers, never
  constants `[VERIFY: numbers moved twice in 2026]`.

### 4.3 Order state machines (both venues, fail-closed)

Per ADR-0005, the OMS state machines encode the venues' documented ambiguity
vocabulary rather than inventing one:

- Kalshi `Text<58>`/reject semantics: `EXCHANGE_UNAVAILABLE` → outcome
  unconfirmed → freeze the order slot, reconcile by `client_order_id`/ClOrdID
  (same-ID retry is the documented recovery), never blind-resubmit;
  `INTERNAL_ERROR` → definitely-not-applied → slot released; post-only cross,
  FOK insufficiency, self-cross, insufficient balance → terminal rejects with
  distinct reasons preserved in the journal.
- Polymarket: order `live` → `MATCHED` (economic fill) → `MINED` → `CONFIRMED`
  | `RETRYING` → `FAILED` (position unwind event). Unknown/timeout states
  freeze the (token, side) slot pending `GET /data/order/{id}` + user-WS
  reconciliation.
- Both: every transition is an append-only journal event with the venue's raw
  payload content-addressed alongside (ADR-0004); an unknown state blocks new
  exposure on that market only (bounded blast radius), consistent with the
  charter's fail-closed decision.

---

## 5. Architecture

### 5.1 Component diagram

```
                        ┌──────────────────────────────────────────────────────────┐
                        │              CONTROL PLANE (slow path, existing)          │
                        │  discovery/metadata · canonical IDs · experiment registry │
                        │  venue-rule snapshot capture · replay/backtest · reports  │
                        └───────────────▲──────────────────────────▲───────────────┘
                                        │ versioned contracts       │ journal reads
                                        │ (rule snapshots, IDs)     │
┌───────────────────────────────────────┴───────────────────────────┴──────────────┐
│                        EXECUTION PLANE (hot path, per venue-region PoP)          │
│                                                                                  │
│  Kalshi WS ──► kalshi-md ──┐                                   ┌─► kalshi-fix ──► mm.fix…:8228
│  (capture tee──►raw store) │   ┌──────────┐   ┌────────────┐   │   (KalshiNR, CoD=Y)
│                            ├──►│ book core│──►│  strategy  │   │
│  PM WS ──────► poly-md ────┘   │ (arrays, │   │  runtime   │   ├─► kalshi-rest ─► external-api…
│  (capture tee──►raw store)     │ seq-gap  │   │ (pre-reg   │   │   (V2 event orders)
│                                │ fail-    │   │  signals,  │   │
│                                │ closed)  │   │  XGBoost   │   └─► poly-clob ───► clob.polymarket.com
│                                └──────────┘   │  native)   │       (EIP-712 sign pipeline,
│                                               └─────┬──────┘        heartbeats, user-WS fills)
│                                                     │ OrderIntent
│                                               ┌─────▼──────┐
│                                               │ risk gate  │  position/notional/price-band/
│                                               │ (in-line)  │  rate budget/kill switch;
│                                               └─────┬──────┘  venue rails: order groups,
│                                                     │          heartbeats, CoD
│                                               ┌─────▼──────┐
│                                               │    OMS     │  single-writer per (venue,market);
│                                               │ state mach.│  ClOrdID discipline; fail-closed
│                                               └─────┬──────┘  reconciliation
│                                                     │
│         append-only journal ◄────────────────────────┘  every inbound/outbound event,
│         (existing raw/canonical stores)                 content-addressed, replayable
└──────────────────────────────────────────────────────────────────────────────────┘
```

One process per venue PoP contains md-adapter(s), book core, strategy runtime,
risk gate, OMS, and execution adapter as components on one event loop —
in-process function calls and ring buffers, no brokers, no RPC (honors the
no-microservices NO-GO and keeps replay deterministic). The capture tee keeps
the existing raw-first recording exactly as it is today; the live book is a
consumer of the same frames, so recorded evidence and live state cannot
diverge silently.

### 5.2 Mapping to accepted ADRs

| ADR | How this design complies |
|---|---|
| 0001 (engine deferred) | Everything here is venue adapters + versioned contracts; no framework types cross the boundary. NautilusTrader remains a candidate *engine* evaluated on X-09-class evidence later; its Polymarket adapter meanwhile serves as reference material only. |
| 0002 (native adapter boundary) | kalshi-md/kalshi-fix/kalshi-rest/poly-md/poly-clob own wire protocol, auth, sequence, reconnect, rate limits, raw capture, native order state. Control plane owns canonical IDs, envelopes, replay, simulation. |
| 0003 (hot path) | Hot path = capture → normalize → book → pre-registered signals → risk → order state → journal. No discovery, no LLM, no PMXT in the loop. Model artifacts are pre-registered, versioned, loaded at start. |
| 0004 (event sourcing) | Every venue frame and every OMS transition appends raw + canonical events; live state is reconstructable; TCA runs off the journal. |
| 0005 (fail closed) | Unknown order state freezes the slot; missing/stale rule snapshot blocks the market; feed seq gap invalidates the book until resnapshot; no default fees/delays ever. |

### 5.3 Book core

Prediction-market books live on a bounded price grid ((0,1) with per-market
step, at most a few thousand levels even at $0.0001 ticks). The book is a flat
array indexed by tick with best-bid/ask cursors — O(1) updates, cache-friendly,
trivially snapshottable for the journal, and identical in backtest and live.
Kalshi's `price_ranges` bands and Polymarket's dynamic tick size arrive as
lifecycle events that re-grid the book deterministically. Sequence handling:
Kalshi per-subscription `seq` contiguity, Polymarket book-hash/snapshot
reconciliation; a gap marks the book stale (fail-closed) and triggers
`get_snapshot`/resubscribe while the risk gate blocks new intents on that
market.

### 5.4 Strategy runtime and models

The runtime consumes normalized book/trade/lifecycle events plus registered
game-state events, evaluates pre-registered signal functions (the existing NFL
model line: XGBoost inference via the native C API on preallocated buffers;
µs-scale), and emits `OrderIntentV0`s. Determinism contract: same canonical
event sequence in, same intent sequence out — which makes shadow mode (intents
scored against the existing X-07 taker simulator on live books) a valid
pre-live rehearsal of literally the production code path, and makes replay
Level 1/2 checks applicable to the execution plane.

### 5.5 New contracts (extend `contracts.py`, versioned v0)

- `OrderIntentV0` — strategy output: venue, market/condition, side (yes-leg
  vocabulary), limit price (FixedPoint), quantity, TIF, post-only, STP class,
  intent lineage (signal id + input event hashes).
- `OrderStateV0` / `ExecutionEventV0` — OMS state-machine records: native IDs
  (ClOrdID / order hash), full native transition vocabulary incl.
  `EXCHANGE_UNAVAILABLE`-class unknowns, fees as venue-reported values,
  Polymarket settlement sub-states, Kalshi shard id.
- `VenueRuleSnapshotV0` extensions: `price_ranges`/tick structure,
  `exchange_index`, fee `fee_type`/`fee_multiplier`/category rate, maker-fee
  flag, delay fields per order class (the existing `seconds_delay` generalizes
  to `{taker_delay, sports_in_play_delay, …}` — captured from venue docs/API,
  content-addressed, as-of joined), rate-limit tier snapshot.
- `TcaRecordV0` already carries markouts and fee lineage; live fills reuse it
  unchanged, which gives sim-vs-live drift measurement for free.

### 5.6 Runtime/language decision (with escape hatch)

**v0 is Python 3.12 in this repo** (asyncio + uvloop; `cryptography` for
RSA-PSS; `coincurve` for secp256k1; `xgboost` native predict; optional
Cython/Rust kernels later). Rationale: the venue floor is ~10–20 ms RTT and
the internal budget (§3.1) totals well under 1 ms in Python if allocation-
disciplined (preallocated buffers, `gc.freeze` + tuned/disabled cyclic GC on
the hot loop, no reflection in codecs); the whole governed determinism harness
(contracts, replay, simulator) is already Python; and the charter forbids
premature infrastructure sprawl. **Escape-hatch criteria (registered
experiment, X-09-style):** if measured p99.9 internal latency exceeds 2 ms or
GC pauses > 5 ms appear at realistic burst rates (recorded game-day replay at
≥10× speed), port the book core + codecs + signing pipeline to a single Rust
extension crate (PyO3) behind the same contracts — not a rewrite, and
explicitly not a framework adoption (ADR-0001 untouched). The measurement
harness itself lands in Phase 1 so this trigger is evidence, not vibes.

---

## 6. Open-source: adopt / reference / avoid

Survey date 2026-08-17; licenses and maintenance verified per §10 sources.

**Adopt (as libraries, never as canonical-record producers):**

| Project | Role | Notes |
|---|---|---|
| hftbacktest (MIT, Rust+Py) | Backtesting spine for latency/queue-aware execution research | Only OSS with feed/order latency models + queue-position fills + L2/L3 replay. We write Kalshi/PM → hftbacktest converters from our raw captures; live side not used |
| Polymarket `py-sdk` / `py-clob-client-v2` (MIT) | Reference + utility for V2 auth/signing; source of typed-data definitions | Wrap, don't trust: our adapter owns transport, retries, journal. Rust path: official `rs-clob-client-v2` + alloy |
| `coincurve`/libsecp256k1, `cryptography` | Signing primitives | Precomputed EIP-712 hashes; warmed contexts |
| QuickFIX-family engine (QuickFIX/J, `quickfix` bindings, quickfix-rs) | Fallback FIX engine if in-house FIXT initiator underperforms conformance | Kalshi docs name QuickFIX-family compatibility; dictionary XML published |
| uvloop, (later) PyO3+tokio stack | Runtime substrate | io_uring runtimes and kernel bypass are the wrong problem for cloud venues |

**Reference (mine for design; do not run):**

- NautilusTrader's Polymarket adapter + docs (LGPL-3.0) — the best public spec
  of V2 operational reality: rate-bucket tiers, four signature types,
  reconciliation drift, WS caps, heartbeat integration. Also the concrete
  engine candidate if ADR-0001's later evaluation selects one; adopting it
  would mean writing its missing Kalshi adapter and accepting LGPL + framework
  coupling — a decision explicitly *not* made here.
- TexasCoding `kalshi-python-sdk` — de-facto documentation of all Kalshi FIX
  session types in working Python; conformance oracle for our initiator.
- `taetaehoho/poly-kalshi-arb` (no license → ideas only) — dedup-guarded
  concurrent legs, cross-venue mapping cache, circuit breakers.
- `rodlaf/KalshiMarketMaker` (Avellaneda-Stoikov on Kalshi), Polymarket
  `poly-market-maker` + `warproxxx/poly-maker` (quoting bands) — maker-phase
  strategy references.
- `pmxt` — steal the cross-venue market-equivalence schema; reject the sidecar.
- `barter-data` — clean Rust normalized-stream pattern if/when the Rust
  escape hatch opens.

**Avoid:** fefix/FerrumFIX (dead since 2021) and rustyfix (unstable);
Hummingbot (1 s clock, no relevant connectors); any V1-era Polymarket code
(pre-2026-04-28 — cannot trade on today's exchange); the Kalshi↔PM arb-bot
long tail (unverified "SIMD/lock-free/sub-ms" marketing, several
credential-hungry portfolio clones); pmxt's sidecar on the hot path; fuzzy
text-similarity market matching as an execution trigger (settlement-rule
mismatch is invisible to string similarity — equivalence must be a curated,
versioned registry with explicit settlement-source assertions).

**Build (nothing usable exists):** production Kalshi FIX gateway; Kalshi L2
historical dataset (our recorder is the moat — start now); settlement-aware
cross-venue equivalence registry; deterministic fail-closed OMS to this
program's contracts.

---

## 7. Delivery plan (all gates before real money; charter amendment required to activate)

**Phase 0 — Facts + recording (1–2 weeks, no new trading surface).**
Verification checklist §9.1 executed and filed as venue-connectivity artifacts;
L2 recording extended to target markets on both venues (existing recorder,
new tickers; retention + manifest discipline unchanged); Kalshi demo
credentials provisioned (existing fail-closed gate `artifacts/venue-connectivity/
kalshi_credentials.md`); Kalshi self-serve Advanced tier upgrade; Polymarket
L1→L2 credential mint against a zero-value wallet; hftbacktest converter
prototype over one recorded game day.

**Phase 1 — Execution contracts + shadow OMS (2–4 weeks).** `OrderIntentV0`,
`OrderStateV0`, `ExecutionEventV0`, rule-snapshot extensions; book core +
strategy runtime on live feeds in shadow mode (intents → X-07 simulator, TCA
journaled); risk gate with kill switch; latency measurement harness (p50/p99/
p99.9 per §5.6); replay determinism evidence registered (X-09-class, execution
plane). Exit gate: two identical replays of a full recorded game day produce
byte-identical intent + simulated-fill streams, and shadow TCA is reviewed.

**Phase 2 — Kalshi demo-environment live loop (2–3 weeks, demo funds only).**
kalshi-rest adapter live against `external-api.demo.kalshi.co` with order
groups + cancel-on-disconnect analogues; then the FIX initiator against
`fix.demo.kalshi.co` (KalshiNR, CoD=Y) with conformance tests vs the dictionary
XML; reconciliation drills (kill the session mid-order; verify fail-closed
freeze + recovery); demo-order TCA vs simulator drift report. Polymarket has no
demo venue: its adapter reaches "connected read-only + signed-order dry-run"
(orders constructed + signed, `POST` withheld) plus heartbeat/cancel-all drills
against the live API with zero balances. Exit gate: program review of drift +
reconciliation evidence.

**Phase 3 — Program decision point (calendar-gated by charter, not by code).**
Promotion gates X-01/X-02/X-05/X-09 + Team I compliance green, then a charter
amendment vote on unblocking micro-capital, single-venue, taker-only live
pilot (position/loss caps enforced by order groups + risk gate; Polymarket
heartbeats mandatory). TCA-vs-simulator drift is the pilot's primary output —
it either validates the X-07 economics or halts the program honestly.

**Phase 4 — Post-pilot (each item its own gate).** Maker strategies (currently
NO-GO) with queue-position telemetry; Kalshi tier progression → Premier → FIX
production + PrivateLink; Polymarket maker-volume tier + rebate optimization;
Rust kernels if the §5.6 trigger fires; Polymarket US / QCX FIX as a possible
third integration after Team I review (separate exchange, Ed25519 + FIX,
FCM-intermediated access).

Charter-prohibited items this plan deliberately does not schedule:
multi-venue simultaneous live arbitrage, copy trading, LLM hot path, RL,
microservices decomposition.

---

## 8. Risks

| Risk | Exposure | Mitigation |
|---|---|---|
| Venue rule churn mid-build (sharding completes 08-24; fee changes 08-19/21) | Adapters built on stale semantics | Rule snapshots as streamed data; changelog watch as a standing chore; contracts carry `exchange_index`/fee fields from day one |
| Region facts wrong (both community-sourced) | Misplaced PoPs, wasted spend | §9.1 verification before any placement decision; PrivateLink provisioning reveals Kalshi's region authoritatively |
| Kalshi FIX access below Premier denied | Order path stays REST (≈1 ms signing penalty + no CoD) | REST V2 path is fully specified above and sufficient for pilot volumes at Advanced tier (~30 orders/s); FIX remains an optimization, not a dependency |
| Polymarket V2 `timestamp` freshness window undocumented | Pre-signing design invalid | Treat as sign-at-send until §9.1 empirics; signing budget already ≤150 µs so pre-signing is an optimization, not a requirement |
| No Polymarket demo venue | First live PM order is a real order | Dry-run signing + zero-balance drills + micro-size gated pilot with heartbeats and cancel-all rehearsed |
| Sports in-play delay ambiguity (1 s recorded 2026-07; 500 ms taker delay removed 2026-02) | Entire latency investment mis-calibrated | §9.1 item 1; per-market delay fields in rule snapshots; simulator already consumes `seconds_delay` |
| Fee schedule drift vs hard-coded assumptions | P&L model silently wrong | Fees only via rule snapshots + venue-reported fill fees; TCA cross-checks reported vs computed |
| LGPL exposure if NautilusTrader later adopted | License obligations | Decision deferred (ADR-0001); reference-only use today; Team I review before any adoption |
| Credential/key handling at PoPs | Key theft = fund loss (PM keys move money) | Existing fail-closed credential references; per-venue keys with minimal scopes (Kalshi scoped API keys; PM proxy-wallet signer separation); no keys in repo/logs; kill switches on both venues |
| Compliance/eligibility (venue ToS, jurisdictions, sports-contract regulatory flux) | Program-level | Team I owns; Phase 3 gate is explicit; server placement ≠ legal jurisdiction — flagged for review |

## 9. Verification checklist (Phase 0 exit criteria)

Each item lands as a dated venue-connectivity artifact with raw evidence.

1. **In-play sports delay, per venue, per market class** — Kalshi: market/series
   metadata + docs + a demo-env marketable order timing test; Polymarket: docs
   + observed `book`→trade timing on an in-play market. Decides §3.2 posture.
2. **Regions** — traceroute/latency matrix against `mm.fix.elections.kalshi.com`
   (NLB, not CDN) and PM origin candidates from multiple AWS regions; ask
   institutional@kalshi.com which PrivateLink region(s) are offered.
3. **Kalshi fee PDF vs API** — reconcile `fee_type`/`fee_multiplier`/change
   schedules against the published schedule; snapshot both.
4. **Polymarket live rate limits + `timestamp` freshness window** — read live
   docs/headers; empirically probe signed-order staleness against a zero-value
   order (rejected-for-balance is fine; rejected-for-staleness is the datum).
5. **Kalshi FIX access path below Premier** — institutional@ inquiry for demo
   + prod grants; document the answer.
6. **`ticker_v2` WS channel existence** (docs/AsyncAPI mismatch) and PM GTD
   minimum expiry (60 s vs 3 min).
7. **Kalshi sharding state as of build start** — which categories on which
   `exchange_index`, per-shard `exchange/status` semantics observed.

## 10. Source register (checked 2026-08-16/17)

Kalshi (official): docs.kalshi.com — `getting_started/{api_environments,
api_keys, rate_limits, exchange_sharding, subpenny_pricing,
fixed_point_migration, maintenance_and_pauses, order_groups, market_settlement,
market_lifecycle, fee_rounding}`, `fix/{connectivity, authentication,
order-entry, market-data, drop-copy, listener-sessions, session-management,
changelog}`, `api-reference/*`, `changelog`, `openapi.yaml`, `asyncapi.yaml`,
FIX dictionary `assets.kalshi.com/fix/kalshi-fix-dictionary.xml`. (Read via the
nightly-synced mirror `github.com/ammario/kalshi-docs`, sync date 2026-08-16.)

Polymarket (official): docs.polymarket.com (auth, order lifecycle, heartbeats,
RTDS, v2-migration, rate limits), github.com/Polymarket/{py-clob-client-v2,
py-sdk, rs-clob-client-v2, clob-client-v2, ctf-exchange-v2 (V2 contracts +
audits), agent-skills, real-time-data-client}; help.polymarket.com (2026-04-28
exchange upgrade; maker rebates). Polymarket US: docs.polymarket.us
(institutional FIX overview, FCM pages).

Third-party (flagged where load-bearing): NautilusTrader Polymarket adapter
docs (rate/tier snapshot 2026-08-04); QuantVPS + Glassnode latency monitor
(Kalshi us-east-2 attribution `[unofficial]`); Entriken + VPS-vendor guides
(PM eu-west-2 attribution `[unofficial]`); EventWaves/TradingVPS Dublin
measurements `[unofficial]`; fee trackers (Crypticorn, StartPolymarket,
MarketMath) `[reconcile per §9.3-4]`; OSS repos as linked in §6.

## Appendix A — Draft ADR text for Team A (Proposed, not filed)

> **ADR 0007 (draft): Execution-plane protocol selection per venue.**
> Kalshi order entry targets FIX (FIXT.1.1/FIX50SP2, KalshiNR,
> cancel-on-disconnect) with REST V2 event-orders as the fully-supported
> fallback and initial production path; Kalshi market data targets the
> authenticated WebSocket with per-subscription sequence validation. Polymarket
> targets CLOB V2 REST order submission with a precomputed EIP-712 signing
> pipeline, the user WebSocket for execution events, market WebSocket for data,
> and the heartbeat dead-man switch armed whenever orders are open. All venue
> rules (ticks, fees, delays, shard topology, rate tiers) enter execution and
> simulation only as versioned snapshots. This ADR does not select a trading
> engine (ADR-0001) and authorizes no blocked scope.

## Appendix B — Program NO-GO preserved

This proposal authorizes none of the blocked scope: real-money execution;
maker queue strategies or exact queue-fill claims from PMXT L2; multi-venue
live arbitrage; live copy trading; an LLM hot path; reinforcement learning;
large-scale microservices; F1 or MLB productionization; a self-built AMM;
on-chain market making; strategy selection from README return claims; or
unregistered fast backtests.
