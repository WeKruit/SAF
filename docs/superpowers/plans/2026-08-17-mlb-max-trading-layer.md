# MLB Max — Trading-Layer Build, Test & Doc-Watch Plan (v0)

Date: 2026-08-17
Status: **PROPOSAL** — pending program review (Team A + H). Registered as
ART-B-009. Executes nothing until approved.
Owners: B (build) + D4 (MLB strategy consumer) + H (validation protocol).
Upstream: ART-A-021 `artifacts/architecture/trading_layer_design_v0.md`
(design rationale, venue facts as of 2026-08-16/17, source register §10);
charter v0.2.

"MLB Max" names the MLB execution-readiness track: build the Kalshi +
Polymarket trading layer Python-first, advance both venues in lockstep,
test to the submission boundary, run a dual-venue shadow campaign on live
MLB markets, and keep a standing watch on both venues' documentation.
The term is introduced by this plan; the repo's existing MLB assets
(`mlb_game_state.py`, `mlb_season_census.py`, Retrosheet pipeline, D4's
24-state Markov baseline) are its strategy-side consumers.

## 0. Position and hard gates

1. Python-first (uvloop) per ART-A-021 §5.6. The Rust escape hatch stays
   defined but is triggered only by M3 measured p99/p99.9 — never
   pre-emptively.
2. This plan authorizes **build + test + shadow only**. Charter v0.2's MVP
   not-do list explicitly blocks maker execution, simultaneous multi-venue
   production execution, and F1/MLB productionization; ART-A-021 Appendix B
   preserves the program NO-GO on real-money execution. Nothing below
   activates any of these. The plan's purpose is to produce the evidence
   package on which Team A + an X-03 re-scoring could later authorize them.
3. NBA moneyline remains the charter's default production mainline. This
   plan does not contest that; it makes the MLB case measurable.
4. Season clock (the only calendar-hard constraint): 2026 MLB regular
   season runs through late September; postseason (peak liquidity) is
   October. The M3 shadow window targets the September tail + postseason.
   Slipping past October forfeits the MLB window until 2027-04; the
   fallback shadow universe is NBA (season opens late October), which is
   charter-mainline anyway.

## 1. Workstreams and the lockstep rule

- **WS-core** — venue-agnostic: contracts v0, journal, book core, OMS,
  risk gate, strategy runtime port (~75% of code, per ART-A-021 §5).
- **WS-K** — Kalshi adapter: REST V2 + WS recorder + demo-env order loop +
  FIXT initiator (narrow subset).
- **WS-P** — Polymarket adapter: CLOB V2 REST + market/user WS recorders +
  precomputed EIP-712 signing pipeline + heartbeat loop.
- **WS-test** — the tier system of §3, applied at every milestone.
- **WS-watch** — the venue-doc watch routine of §4.

**Lockstep rule:** a milestone closes only when BOTH venue tracks pass the
same test tier for that milestone, and neither venue track may lead the
other by more than one open milestone. Rationale: the shared core must not
be shaped by whichever adapter happens to land first; the asymmetries that
matter (Kalshi has a demo env, Polymarket does not; Kalshi has FIX,
Polymarket does not) are handled by defining per-venue gate equivalents,
not by letting one side run ahead.

## 2. Milestones

### M0 — Foundations, recorders, watch armed (weeks 1–2)

1. Contracts v0: Order/Fill/BookDelta/MarketId types + the per-venue
   ambiguity taxonomy (which errors mean "state unknown") from ART-A-021
   §4.3; append-only hash-chained journal; config/secrets layout; CI
   harness for tiers T0/T1.
2. Kalshi: REST V2 read client; WS L2 recorder with sequence-gap
   accounting; demo-env credentials provisioned, login smoke test green.
   Note: exchange sharding rollout completes 2026-08-24 — recorder capture
   schema is versioned so pre/post-sharding captures stay comparable.
3. Polymarket: market-WS + user-WS recorders; official `py-clob-client-v2`
   wrapped read-only; EIP-712 vector harness that reproduces the official
   client's signing test vectors **byte-identically** (our precompute
   pipeline must match before it may diverge internally).
4. Doc-watch armed per §4; first sweep report filed.
5. Close every ART-A-021 §9 Phase-0 verification item that needs no
   capital: PoP candidate → venue RTT survey; MLB in-play delay
   observation protocol drafted (execution needs live games — September).

Exit evidence: 72h continuous dual-venue capture with gap/duplicate
statistics filed as an artifact; signing vectors byte-identical; docwatch
sweep #1 filed. Lockstep gate: both recorders reach 72h.

### M1 — Read path (weeks 3–5)

1. Book core: normalized L2 delta application, gap→resync policy,
   BBO/depth/imbalance features; property tests against naive full
   rebuild.
2. Replay determinism: recorded streams replayed twice → identical journal
   hashes, both venues (repo determinism culture; same standard the
   game-state engine met).
3. Stage-separated latency bench harness with deterministic report format
   (reuse the game-state engine's bench conventions).
4. MLB market discovery: enumerate both venues' MLB series/events daily;
   seed the cross-venue equivalence registry — curated and
   settlement-source-asserted, never fuzzy text matching (ART-A-021 §6).

Exit evidence: replay-determinism artifact for both venues; equivalence
registry v0 dual-mapping every live MLB game in the capture window; bench
baseline report.

### M2 — Order path to the submission boundary (weeks 5–8)

1. OMS: fail-closed state machine including unknown/ambiguous states,
   idempotency keys, reconciliation loop, mass-cancel-on-ambiguity drill.
2. Risk gate: notional/position/rate caps, kill switch, venue dead-man
   integration (Polymarket heartbeat; Kalshi session loss ⇒ cancel-on-
   disconnect posture).
3. Kalshi: demo-env full order loop green over REST V2
   (place/cancel/replace/fill/reconcile), then 24h soak with injected
   disconnects. FIXT initiator development starts (session layer +
   NewOrderSingle/Cancel + ExecutionReport only), conformance-rigged
   against TexasCoding `kalshi-python-sdk` as oracle; FIX completion may
   trail into M3 — the lockstep gate counts REST V2, FIX is Kalshi-only
   surplus.
4. Polymarket: full pipeline to the submission boundary — precomputed
   struct-hash cache keyed to the price ladder, warmed secp256k1 contexts,
   heartbeat loop in an isolated process — then the **unfunded-reject
   conformance probe**: submit from a zero-balance wallet; the pass signal
   is the venue rejecting for insufficient balance/allowance and NOT for
   invalid signature/auth. This exercises auth, signing, and transport
   with no capital at risk. Gated on Team I confirming O-001 (Polymarket
   ToS) permits it.
5. Chaos tier first pass: WS kill mid-stream, heartbeat starvation, clock
   skew, GC-pressure runs; verify fail-closed behavior every time.

Exit evidence: exhaustive state-machine transition evidence (property
tests); Kalshi demo soak report; Polymarket boundary conformance report;
chaos report. Lockstep gate: Kalshi demo loop ≈ Polymarket reject-probe
(the per-venue T2 equivalents).

### M3 — Dual-venue shadow on live MLB (weeks 8–10, September window)

1. Strategy runtime port + D4 consumer stub (game-state → intent); shadow
   OMS runs BOTH venues concurrently on live feeds — full pipeline, orders
   journaled at the submission boundary, never submitted. Kalshi demo env
   may run live in parallel as the one true-loop reference.
2. Tick-to-ack p50/p99/p99.9 measured per stage under live load, GC tuning
   applied (`gc.freeze` + gen2 discipline per the performance analysis);
   this produces the Rust-escape-hatch decision input.
3. MLB in-play delay measured from recorded books around scoring events —
   the decision-critical ART-A-021 [VERIFY] item, answered for MLB
   specifically.
4. Assemble the Team A + H review package: everything above plus the
   docwatch delta digest since M0.

Exit = review package registered. **M4 (micro-capital, maker, MLB
productionization) is explicitly not entered** — it requires a charter
amendment plus X-03 re-scoring, per §0.

## 3. Test tiers (WS-test)

- **T0 unit/property** (per-commit CI): OMS state machine exhaustive
  transitions (hypothesis); book-delta properties vs naive rebuild;
  EIP-712 vectors pinned to the official client release; Kalshi FIX golden
  messages validated against the published dictionary XML.
- **T1 record/replay** (per-commit CI): deterministic journal hashes;
  injected gap/duplicate/reorder cases.
- **T2 venue boundary** (nightly; needs demo creds + unfunded wallet as CI
  secrets): Kalshi demo-env loop; Polymarket unfunded-reject probe.
- **T3 latency bench** (nightly): stage-separated distributions,
  deterministic report; trend tracked across commits.
- **T4 chaos/liveness** (weekly): disconnect storms, heartbeat starvation,
  GC pressure, mass-cancel drills.

House rules: no flake quarantine — a failing test blocks the milestone; a
docwatch release alert on either official client opens a re-pin task for
the vector suite; bench regressions >20% at p99 block merge.

## 4. Doc-watch routine (WS-watch)

Standing weekly watch over both venues' documentation, seeded from the
ART-A-021 §10 source register.

| ID | Source | Breaking-class triggers |
|---|---|---|
| W-K1 | Kalshi API docs root + changelog | endpoints, auth, order semantics |
| W-K2 | Kalshi FIX spec + dictionary XML | any dictionary or session change |
| W-K3 | Kalshi rate-limit / tier page | tier thresholds, new limits |
| W-K4 | Kalshi fee schedule | fee formula/coefficients |
| W-K5 | Kalshi status + announcements | sharding, maintenance, delays |
| W-P1 | Polymarket docs root + changelog | endpoints, auth, order semantics |
| W-P2 | `clob-client-v2` / `py-clob-client-v2` / `rs-clob-client-v2` releases | signing struct/domain changes |
| W-P3 | Polymarket fee page | category fees, taker/maker terms |
| W-P4 | Polymarket contract addresses | exchange/adapter address changes |
| W-P5 | Polymarket status page | heartbeat/dead-man rule changes |

Mechanism: `tools/docwatch.py` — fetch each source, normalize (strip
navigation/boilerplate), store SHA-256 + ETag/Last-Modified in
`var/docwatch/state.json`; on change, write a diff report to
`artifacts/docwatch/YYYY-MM-DD.md` and classify **breaking**
(auth/endpoint/signing-domain/rate/fee/delay-rule changes) vs
**informational**. Client releases checked via the GitHub API.

Cadence: weekly full sweep (Mondays 09:00 UTC) run by a scheduled agent
session, plus an on-demand sweep before each milestone close. A breaking
finding opens a GitHub issue and blocks the affected milestone until
triaged; informational findings roll into the weekly digest. The Routine
is armed at M0 and stays on for the life of the trading layer.

## 5. Cadence and reporting

Weekly checkpoint: lockstep gate status, docwatch digest, bench trend,
registry updates. Every milestone close files its evidence as registered
artifacts under the existing discipline (this plan is ART-B-009).

## 6. Dependencies and risks

- **Team I gates**: O-001 (Polymarket ToS) before the reject probe; O-003
  (Kalshi agreement) before demo-env automation.
- **Provisioning**: Kalshi demo account; Polymarket wallet (deliberately
  unfunded); CI secrets for T2.
- **Season clock**: M3 slip past October ⇒ MLB window closes until
  2027-04; fallback is NBA shadow (charter mainline).
- **Sharding rollout** (Kalshi, completes 2026-08-24) lands mid-M0:
  recorders version their capture schema; docwatch W-K5 covers.
- **No Polymarket sandbox**: boundary-testing ceiling acknowledged; the
  first real submission risk concentrates at gated M4, outside this plan.
- **Single-maintainer oracle**: TexasCoding SDK is a conformance oracle,
  never a dependency; if it dies, the dictionary XML + demo env suffice.

## 7. Verification

- targeted tests after each module;
- complete repository test suite;
- deterministic replay twice with identical hashes;
- `tools/validate_governance.py` + registry/program audit;
- clean Git state; push the exact verified commit.

## Appendix — charter cross-references

- MVP not-do items preserved verbatim (charter v0.2 §Team A): maker
  execution, simultaneous multi-venue production execution, LLM in hot
  path, RL, F1/MLB productionization, self-built AMM, on-chain
  market-making, microservices.
- Experiment linkage: X-01 (determinism), X-03 (market census — re-scores
  sport priority), X-04 (event reaction), X-07 (cost floor) consume this
  plan's captures and bench outputs.
- NO-GO on real-money execution: ART-A-021 Appendix B, unchanged.
