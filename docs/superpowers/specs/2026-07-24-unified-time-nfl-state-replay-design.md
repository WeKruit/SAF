# Unified Time and NFL State Replay Design

Date: 2026-07-24

Status: approved in conversation; awaiting written-spec review

Program status: CONDITIONAL_GO

Initial vertical slice: `2025_14_DAL_DET`

## 1. Decision

SAF will establish one temporal model for both game data and market data before
performing any event-to-market analysis.

The two source streams remain independent:

```text
Game stream
G0 source event
  -> G1 source publish/send, when documented
  -> G2 local receive
  -> G3 canonical game state ready

Market stream
M0 exchange event
  -> M1 venue publish/send, when documented
  -> M2 local receive
  -> M3 canonical market state ready

Validated join
J = validated G stream x validated M stream
```

Availability is not a third raw stream. Game availability is `G3`; market
availability is `M3`. The primary future reaction metric is:

```text
first qualifying M3 probability or quote change - G3
```

The initial implementation finishes and validates the NFL game-state replay
before it performs `J`. It does not train a model, estimate alpha, infer
execution quality, or make a causal market-reaction claim.

`G0` means the source-reported occurrence of a game event. It is not promoted
to physical ground truth unless an independent source proves that semantic.

## 2. Program boundaries

This design inherits, without changing, the Charter's approved architecture,
deterministic-replay requirements, experiment registry discipline, and NO-GO
list.

It instantiates the existing event-sourcing and fail-closed architecture. It
does not change a high-level architectural decision and therefore does not
trigger a new ADR.

Included:

- unified time semantics and uncertainty intervals;
- operational capture timing for game and market streams;
- deterministic NFL state replay;
- official Game Book reference-only validation;
- historical and prospective time-quality reporting;
- market-implied probability normalization as the next join input.

Excluded:

- real-money activity;
- maker or queue modeling;
- live or historical execution claims;
- training or evaluating game-state probability models;
- using market movement to repair the game clock;
- treating historical archive download time as historical availability;
- LLM use in any hot path or as the final validation authority.

## 3. Existing evidence preserved

The first state-replay target remains `2025_14_DAL_DET`.

Current accepted evidence:

- frozen nflverse source: 192 rows;
- 191 deterministic state transitions;
- 187 source `time_of_day` values and 5 administrative rows without one;
- final score: Detroit 44, Dallas 30;
- two-run canonical trace hash:
  `sha256:f44a1bd1e42be8b675e4ba4aeca5f276cf03c641f120d56979eacc5055695f09`;
- 23 drives x 9 fields and 15 scoring summaries x 7 fields match the official
  Game Book;
- existing Polymarket and Kalshi historical market captures remain immutable.

The current evidence does not yet prove:

- full 192-row mapping to official Game Book play evidence;
- historical local receipt or source publication times;
- precise point-in-time state revisions during reviews;
- event-to-market alignment;
- reaction latency, causality, alpha, or executability.

Existing raw objects and artifacts are not rewritten. Missing historical
monotonic or local-receive timestamps remain missing.

## 4. Unified temporal evidence

### 4.1 One timestamp shape

Every game or market timestamp is represented as evidence, not as a naked
datetime:

```text
raw_value
role
clock_domain
utc_lower_ns
utc_upper_ns
precision_ns
semantics_status
application_point
source_sequence
clock_sync_ref
lineage_ref
```

Allowed `semantics_status` values:

```text
verified
inferred
ambiguous
unavailable
```

All normalized UTC intervals are half-open:

```text
[utc_lower_ns, utc_upper_ns)
```

A second-precision source value represents a one-second interval. A
minute-ending candle represents its complete source-defined minute. Missing
precision is not invented.

### 4.2 Local operational timing sidecar

Runtime timing is stored in an operational sidecar keyed by:

```text
capture_session_id + record_ordinal + raw_payload_sha256
```

The sidecar does not change canonical event identity or source-object hashes.
Each receive boundary records:

```text
local_receive_utc_ns
local_receive_monotonic_ns
clock_pairing_uncertainty_ns
host_id
boot_id
connection_epoch
clock_sync_sample_id
```

Each normalization boundary records:

```text
parse_done_monotonic_ns
normalize_done_monotonic_ns
state_or_book_ready_monotonic_ns
```

Durations on one host use the monotonic clock. Cross-clock comparisons use UTC
intervals and must include the combined clock uncertainty.

### 4.3 Clock-health evidence

Clock health is an append-only operational stream containing:

```text
clock_sync_sample_id
host_id
boot_id
sample_utc_ns
sample_monotonic_ns
sync_state
reported_offset_ns
reported_error_bound_ns
time_source
```

Allowed `sync_state` values are `LOCKED`, `HOLDOVER`, and `UNSYNCED`.
`UNSYNCED` samples may support same-host monotonic durations only.

If a wall-clock step is detected, the capture is split into a new timing
epoch. Samples never cross a clock step, reconnect gap, or boot boundary.

## 5. Interval construction

### 5.1 Game intervals

Each NFL play stores separate intervals for:

- play start;
- play completion;
- result finalization;
- local receipt, when prospectively captured;
- canonical state ready.

Historical replay runtime is a reproducibility benchmark, not the historical
`G3`. Historical `G3` remains unavailable and can only be represented by a
registered availability scenario. Prospective `G3` is the first point at which
the locally received event has been normalized and committed to canonical game
state.

Construction is evidence-driven:

1. A documented start and end time produce the corresponding precision-bounded
   intervals.
2. If only a start time exists, the next timed source event supplies a
   conservative upper bound and the interval is `inferred`.
3. An interval that is too wide for the requested analysis horizon is
   ineligible rather than narrowed manually.
4. A review creates a provisional event and a later revision. The final result
   is never backdated to the provisional play.
5. Administrative rows remain in the source audit. Missing wall time is never
   interpolated merely to place them on a merged UTC timeline.

For the current historical nflverse replay, `time_of_day` is a
source-reported play time, not a local receipt time. The official Game Book
validates sequence, score, drives, rulings, and game clock, but does not prove
every play's absolute UTC or publication time.

### 5.2 Market intervals

Market observations preserve their native meaning:

- trade execution: point interval expanded by source timestamp precision;
- tick L1 or L2 update: source event interval plus local receive timing when
  captured;
- Kalshi one-minute candle: `[end - 60 seconds, end)`;
- lifecycle event: its documented source interval;
- source with undocumented timestamp application point: `ambiguous`.

Trades, candles, quotes, and L2 events are not collapsed into a generic
`market_price`.

### 5.3 Historical availability scenarios

Historical 2025 data has no contemporaneous `G2`, `G3`, `M2`, or `M3`.
Archive `fetched_at` values are excluded from historical availability.

Historical analysis may use a registered scenario:

```text
simulated_game_available = result_finalization_interval + configured_delay
```

The configured delay is a sensitivity parameter, not an observed latency. It
is versioned in the experiment registry and cannot be altered after inspecting
the result.

Once prospective captures exist, source-specific replay profiles use observed
P50, P95, and P99. Five-, ten-, and thirty-second values remain stress
scenarios, not the default definition of normal latency.

## 6. Latency measurements

The system reports separate latency segments:

```text
game_source_age       = G2 - G0
game_ingest           = G3 - G2
market_source_age     = M2 - M0
market_ingest         = M3 - M2
observed_reaction     = first qualifying M3 - G3
```

`game_ingest`, `market_ingest`, and same-host `observed_reaction` use
monotonic timestamps. `game_source_age` and `market_source_age` are cross-clock
intervals and are not called network latency unless both timestamp application
points and clock uncertainty are proven.

For an observed cross-clock difference `d` with endpoint uncertainties
`epsilon_a` and `epsilon_b`, the reported interval is:

```text
[d - (epsilon_a + epsilon_b), d + (epsilon_a + epsilon_b)]
```

P50, P95, and P99 use a frozen nearest-rank estimator. Interval-valued samples
produce percentile bands by taking the same percentile separately over all
lower and upper bounds.

Every percentile report includes:

- sample count;
- endpoint coverage;
- negative and unresolved counts;
- reconnect, recovery, duplicate, and gap cohorts;
- source, message type, connection epoch, and clock-sync status;
- timeout and censored-observation rates.

The existing prospective Polymarket capture provides an uncorrected
source-timestamp-to-callback diagnostic of approximately P50 57 ms, P95
160 ms, and P99 247 ms over about 70,000 incremental frames. Its negative
values and undocumented payload timestamp application point prevent promotion
to a network-latency result.

## 7. NFL state replay

### 7.1 Inputs

The first replay consumes only:

- the verified nflverse 2025 object and static manifest;
- the verified single-game row selection;
- the current NFL reducer contract;
- the official DAL-DET Game Book as reference-only validation evidence;
- content-addressed Game Book facts with page/span citations.

No market data enters the reducer or game-state validation.

### 7.2 Canonical event

Each normalized play contains:

```text
canonical_game_id
native_play_id
native_order_sequence
event_revision
event_type
event_subtype
description
participants
period
game_clock
play_start_interval
play_complete_interval
result_finalization_interval
pre_state
post_state
source_lineage
quality_flags
event_sha256
```

The replay state contains at least:

- period and game clock;
- home and away score;
- possession;
- down and distance;
- field position;
- drive identity;
- team timeouts;
- play and terminal status.

Participants and detailed play semantics are event attributes. They do not
silently mutate the state when the source evidence is incomplete.

### 7.3 Deterministic reducer

The reducer processes native order, with explicit deterministic tie-breaking
only where source ordering is insufficient. It produces:

- one initial state;
- one transition per accepted state-changing source row;
- pre-state and post-state hashes;
- one canonical trace hash;
- a final-state hash.

Two independent runs must satisfy Charter deterministic replay Level 1 and
Level 2. Binary file equality remains optional Level 3.

### 7.4 Official validation

Official Game Book extraction follows:

```text
PDF bytes
-> deterministic text/table extraction
-> optional slow-path LLM candidate facts
-> citation and schema verification
-> deterministic matching to replay events
-> human review only for unresolved mismatches
```

An LLM cannot produce a PASS decision, alter source data, infer missing UTC,
or move an event in time.

The state-replay gate requires:

- all 192 source rows mapped to an official fact or an explicit administrative
  reason;
- every state-changing row supported by official evidence;
- all scoring, turnover, punt, timeout, no-play penalty, quarter boundary, and
  replay-review invariants reconciled;
- quarter and final scores reconciled;
- every mismatch retained and surfaced;
- two-run canonical hashes equal.

Because the Game Book and nflverse may share an NFL/GSIS upstream, this proves
parser and replay consistency, not complete source independence.

## 8. Market probability input after the state gate

After G passes, M is built independently for each venue and proposition.
Every market observation preserves raw price and a canonical
market-implied-probability proxy:

```text
raw_price
canonical_proposition
probability_proxy
bid_probability
ask_probability
last_trade_probability
vwap_probability
observation_interval
```

Opposite outcomes are transformed only after settlement rules prove the
propositions mutually exclusive and exhaustive. Bid and ask remain separate;
midpoint is not treated as executable evidence.

M must pass identity, pagination, deduplication, timestamp semantics, active
interval, and deterministic replay checks before J is allowed.

## 9. Join gate and future trend output

J consumes only accepted G and M artifacts. It never reparses raw source
objects.

For game availability interval `[L, U)` and market observation interval
`[m0, m1)`:

```text
m1 <= L   definitely_pre
m0 >= U   definitely_post
otherwise overlap_ambiguous
```

Only `definitely_post` observations enter formal event-after trend metrics.
Overlapping minute candles cannot establish sub-minute reaction timing.

The first trend output will report, per venue and proposition:

- absolute probability-proxy path;
- probability-point change from a non-stale, definitely-pre baseline;
- bid and ask paths when admissible;
- volume and trade count;
- time to 50% and 90% of the observed move;
- maximum overshoot and subsequent correction;
- sensitivity to source-specific P50, P95, and P99 availability profiles.

The output remains descriptive. It does not claim fair value, causality,
alpha, or executability.

## 10. Artifacts

The design introduces these derived artifacts:

```text
contracts/temporal-evidence/v0.schema.yaml
contracts/capture-timing/v0.schema.yaml
contracts/game-replay/v1.schema.yaml

artifacts/time/nfl_2025_14_dal_det_time_audit_v0.json
artifacts/game-state/nfl/nfl_2025_14_dal_det_events_v1.jsonl
artifacts/game-state/nfl/nfl_2025_14_dal_det_state_trace_v1.jsonl
artifacts/game-state/nfl/nfl_2025_14_dal_det_state_replay_v1.json
artifacts/game-state/nfl/nfl_2025_14_dal_det_gamebook_audit_v0.json
```

Operational timing sidecars remain outside canonical event identity and bind
to immutable raw records by session, ordinal, and payload hash.

Existing v0 state artifacts remain immutable evidence. New consumers migrate
atomically to the v1 replay contract; no dual-read compatibility layer is
introduced.

## 11. Delivery order and ownership

### Stage 1: temporal contract

Owner: A + C + H

- freeze timestamp roles, intervals, clock domains, uncertainty math, and
  percentile estimator;
- publish temporal-evidence and capture-timing contracts;
- define fail-closed eligibility statuses.

Gate: the same fixtures validate game and market timestamps without
source-specific private fields.

### Stage 2: historical NFL state replay

Owner: D2 + C + H

- upgrade the DAL-DET event and replay artifacts to v1;
- add play completion, finalization, revision, and time-evidence fields;
- complete official Game Book event mapping;
- retain current hashes and artifacts as immutable prior evidence.

Gate: full state-replay and official-reference checks pass, with deterministic
Level 1 and Level 2 hashes.

### Stage 3: prospective timing instrumentation

Owner: C + H

- stamp local receive before parsing;
- record monotonic, UTC, clock-sync, connection, and gap evidence;
- stamp parse, normalize, and state/book-ready boundaries;
- validate current Polymarket recorder timing without reclassifying historical
  records.

Gate: clock-step, reconnect, duplicate, sequence-gap, and unsynced-clock tests
fail closed.

### Stage 4: empirical P95 profile

Owner: C + D2 + H

- capture game and market streams on one host;
- use at least four games and at least 100 uniquely matched state changes for
  the first source-specific profile;
- report P50/P95/P99, sample coverage, interval bounds, and failure cohorts.

The no-paid first profile uses Polymarket Sports WebSocket state changes and
Kalshi `live_data`/`game_stats` polling as separate observed sources. It covers
score, period, status, and documented possession changes, not complete
play-level PBP. A prospective play-level P95 profile remains blocked until a
source with documented play identity and event/update timestamps, such as an
authorized Sportradar feed, is available. This block does not prevent the
historical DAL-DET state replay.

Gate: no cross-clock percentile is promoted without numeric clock uncertainty;
same-host reaction timing uses monotonic timestamps.

### Stage 5: venue replay and J

Owner: C + E + H

- normalize Polymarket and Kalshi observations independently;
- normalize probability direction only after rule and identity validation;
- join accepted G and M artifacts;
- generate the first descriptive probability-trend report.

Gate: no alpha, causality, executable-price, or latency claim exceeds the
available timestamp and quote evidence.

## 12. Acceptance tests

Required tests include:

- timestamp precision expands to the correct half-open interval;
- absent source/publish/receive time remains absent;
- wall-clock steps split timing epochs;
- monotonic durations remain valid across wall-clock offset changes;
- sidecar session, ordinal, and payload-hash mismatch fails;
- historical `fetched_at` cannot enter an availability or latency metric;
- review reversal produces separate provisional and final revisions;
- shuffled input cannot alter canonical replay output;
- missing, duplicate, or conflicting source order fails or receives an
  explicit exclusion reason;
- all DAL-DET state-changing events reconcile to official evidence;
- two replay runs produce identical semantic output and canonical hash;
- interval-overlap observations cannot be classified as definitely post;
- candles cannot produce sub-minute reaction timing;
- unproven outcome orientation blocks probability normalization;
- unregistered availability-profile changes invalidate the result.

## 13. Completion definition

This phase is complete only when:

1. one time contract governs both G and M;
2. DAL-DET state replay passes its deterministic and official-reference gates;
3. historical availability limitations are explicit and machine-enforced;
4. prospective capture can measure same-host G2/G3/M2/M3 timing;
5. P95 is source-specific and adjustable only through a versioned profile;
6. J remains blocked until both upstream streams independently pass.
