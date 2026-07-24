# Unified Time and NFL State Replay Design

Date: 2026-07-24

Status: approved in conversation; awaiting written-spec review

Program status: CONDITIONAL_GO

Initial vertical slice: `2025_14_DAL_DET`

## 1. Decision

SAF will establish one temporal model for both game data and market data before
performing any event-to-market analysis.

The temporal model adopts the parts of traditional HFT timestamp discipline
that are necessary for evidence-grade research: every timestamp names its
clock and sampling boundary, same-host durations use a monotonic clock,
cross-clock comparisons carry numeric uncertainty, clock or connection
discontinuities open new epochs, and no result claims more precision than its
weakest timestamp.

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
- HFT-style timestamp sampling boundaries, clock discipline, latency
  decomposition, and eligibility gates;
- operational capture timing for game and market streams;
- append-only acquisition evidence and externally anchored capture roots;
- deterministic NFL state replay;
- official Game Book reference-only validation;
- fail-closed data, identity, time, and market-reaction audit decisions;
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

- externally anchored existence of existing raw objects at their claimed
  acquisition times;
- that a source publisher's content was factually correct;
- full 192-row mapping to official Game Book play evidence;
- historical local receipt or source publication times;
- precise point-in-time state revisions during reviews;
- event-to-market alignment;
- complete contract, team, and outcome orientation for the DAL-DET market
  observations;
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
source_clock_owner
source_clock_error_lower_ns
source_clock_error_upper_ns
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
receive_boundary
clock_pairing_uncertainty_ns
host_id
boot_id
monotonic_clock_id
clock_epoch_id
connection_epoch
clock_sync_sample_id
```

Each normalization boundary records:

```text
raw_append_done_monotonic_ns
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
sync_protocol
sync_daemon_config_sha256
reference_clock_identity
leap_status
reported_offset_ns
reported_error_bound_ns
time_source
valid_from_utc_ns
valid_until_utc_ns
```

Allowed `sync_state` values are `LOCKED`, `HOLDOVER`, and `UNSYNCED`.
`UNSYNCED` samples may support same-host monotonic durations only.
`LOCKED` is not sufficient by itself: the event must fall inside the sample's
validity interval and the numeric error bound must satisfy the registered
experiment budget. `HOLDOVER` error grows using the registered drift bound
until it either exceeds the budget or a new synchronization sample arrives.

If a wall-clock step is detected, the capture is split into a new timing
epoch. A synchronization-daemon restart, reference-clock change, PTP hardware
clock reset, or leap/smear policy change also opens a new clock epoch. A
connection epoch is separate: samples never cross a clock epoch, reconnect
gap, or boot boundary.

### 4.4 HFT timestamp sampling boundaries

The system distinguishes the time at which an event happened from the time at
which each local processing boundary observed it:

```text
source/exchange event
source publish, when documented
optional NIC hardware receive
optional kernel software receive
application callback entry
raw append complete
parse complete
normalize complete
canonical state or book ready
```

`G2` and `M2` mean application callback entry because this is the first
required boundary at which the SAF process can use the message. The baseline
recorder samples paired UTC and monotonic clocks immediately on callback
entry, before parsing. Raw append completes before parsing and has its own
monotonic timestamp.

The baseline pairing uses a monotonic-UTC-monotonic sandwich:

```text
mono_before
utc_sample
mono_after
```

The UTC sample maps to the monotonic interval
`[mono_before, mono_after]`; half the sandwich width, clock resolution, and
the active synchronization error contribute to the pairing uncertainty. The
raw three samples are retained so the mapping can be recomputed. A nearest
clock sample outside its validity interval is never reused.

Kernel or NIC timestamps are additional evidence. They must be labelled
`kernel_software_receive` or `nic_hardware_receive` and must never be
substituted silently for application receipt. When the transport or WebSocket
library exposes no packet timestamp, the record remains
`application_callback`; nanosecond storage does not imply nanosecond
accuracy.

For TLS and WebSocket transports, a packet timestamp is not automatically a
message timestamp. Kernel or NIC evidence enters a message-level latency only
when a deterministic packet-to-frame mapping exists; otherwise it remains a
transport diagnostic and callback entry remains `G2` or `M2`.

This follows the traditional HFT distinction between exchange time, wire or
kernel arrival, application arrival, and decision-ready time. Linux
`SO_TIMESTAMPING` is the reference receive boundary when later deployments
support kernel or NIC timestamping. PTP hardware clocks are a future stronger
clock source, not a requirement for the current historical replay.

### 4.5 Clock-quality classes

Each timing epoch receives exactly one clock-quality class:

```text
HISTORICAL_SOURCE_ONLY
LOCAL_MONOTONIC_ONLY
UTC_BOUNDED_SOFTWARE_RX
UTC_BOUNDED_HARDWARE_RX
```

- `HISTORICAL_SOURCE_ONLY` supports event ordering and source-time intervals,
  but no observed local availability or latency.
- `LOCAL_MONOTONIC_ONLY` supports durations between boundaries on one
  `host_id + boot_id`, but no cross-clock or cross-host subtraction.
- `UTC_BOUNDED_SOFTWARE_RX` adds a synchronized UTC clock, an application or
  kernel software receive boundary, and a numeric error bound.
- `UTC_BOUNDED_HARDWARE_RX` additionally requires a verified NIC hardware
  timestamp, PTP hardware-clock lineage, and the PHC-to-UTC conversion
  evidence.

The experiment profile declares `max_clock_error_ns`,
`max_clock_sample_age_ns`, and its required class before data is inspected.
Chrony/NTP may satisfy a millisecond-scale research profile when its measured
error bound fits the profile. A sub-millisecond cross-host claim requires
PTP-class evidence; the presence of PTP software alone is not proof that the
clock was synchronized.

External source or exchange clocks have their own evidence. Source timestamps
record the clock owner, documented UTC traceability, application point,
precision, and published error bound. Local clock health cannot establish the
quality of `G0`, `G1`, `M0`, or `M1`. If the external error bound or
application point is unknown, source-to-local differences remain diagnostic
and cannot enter a promoted cross-clock latency percentile.

### 4.6 HFT-derived timing gates

The following rules are fail closed:

1. Same-host processing and reaction durations use monotonic timestamps from
   the same host, boot, monotonic clock ID, and clock epoch.
2. Cross-host, source-to-local, and exchange-to-local differences use UTC
   intervals widened by both endpoint error bounds.
3. `UNSYNCED`, stale clock-health evidence, a wall-clock step, boot change, or
   connection discontinuity ends the eligible epoch.
4. A negative cross-clock duration that is contained within combined
   uncertainty is `UNRESOLVED`; one outside the uncertainty bound is a time
   integrity failure. Neither is silently clamped to zero.
5. A missing or ambiguous publish-time application point prevents a network
   latency claim even if the numeric timestamp parses.
6. Source sequence gaps, duplicate conflicts, or reconnect intervals block
   event-level results that depend on the affected interval. A recovery
   snapshot starts a new segment; it does not backfill the missing path.
7. A resolved latency claim must have an interval narrower than its registered
   error budget. Every percentile discloses its timestamp class, boundary,
   clock profile, interval width, and valid sample count.

Historical scenario delays remain sensitivity parameters. They never upgrade
`HISTORICAL_SOURCE_ONLY` records into observed HFT latency evidence.

Each sequenced feed also records:

```text
sequence_domain
sequence_epoch
expected_sequence
observed_sequence
gap_lower
gap_upper
recovery_snapshot_id
```

A gap invalidates incremental state until a verified recovery snapshot and
subsequent continuous deltas open a new valid segment. Sequence reset, wrap,
and reconnect rules are source-specific contracts; a reconnect alone never
closes a gap.

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

Every observed or simulated availability profile is bound to:

```text
source_id
feed_version
transport
receive_boundary
region
host_class
message_type
capture_period
clock_profile_hash
```

A profile mismatch blocks substitution. In particular, a prospective
WebSocket callback profile cannot be presented as the historical availability
of an archive or a different sports feed.

Once prospective captures exist, source-specific replay profiles use observed
P50 and P95. P99 is reported as a descriptive tail statistic until the
pre-registered minimum tail count is met; 100 samples cannot promote a stable
P99. Five-, ten-, and thirty-second values remain stress scenarios, not the
default definition of normal latency.

## 6. Latency measurements

The system reports separate latency segments:

```text
game_source_processing    = G1 - G0, when G1 exists
game_transport_delivery  = G2 - G1, when G1 exists
game_source_age          = G2 - G0
game_ingest              = G3 - G2

market_source_processing   = M1 - M0, when M1 exists
market_transport_delivery = M2 - M1, when M1 exists
market_source_age         = M2 - M0
market_ingest             = M3 - M2

ingress_queue =
  application_callback - kernel_or_hardware_receive, when proven

observed_reaction =
  first qualifying M3 - G3
```

`game_ingest`, `market_ingest`, and same-host `observed_reaction` use
monotonic timestamps. `game_source_age` and `market_source_age` are cross-clock
intervals and are not called network latency unless both timestamp application
points and clock uncertainty are proven. Missing endpoints leave the
corresponding segment unavailable; they are not inferred from the total.

UTC intervals already include source precision and clock error exactly once.
For intervals `A=[A_L,A_U)` and `B=[B_L,B_U)`, the only subtraction is:

```text
B - A = [B_L - A_U, B_U - A_L)
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

J preserves two different orderings:

```text
economic ordering:
  market source-event M0
  vs game play-start and result-finalization intervals

availability ordering:
  market-ready M3
  vs game-state-ready G3
```

They are never substituted for one another. A delayed pre-event market update
that arrives after `G3` is not a usable pre-event baseline. A market update
with `M0` before finalization but `M3` after `G3` may describe feed lead or
delivery delay; it is not a clean event-after market reaction.

For a prospective clean pre-event baseline:

```text
M0 upper <= play-start lower
and
M3 <= G3
```

For a prospective clean event-after observation:

```text
M0 lower >= result-finalization upper
and
M3 > G3
```

The same-host `M3 - G3` reaction duration uses monotonic time. `M0` versus
game finalization uses UTC intervals and both source-clock uncertainties.
Historical records without `G3` or `M3` can report only source-time ordering
plus a registered simulated-availability scenario. Overlapping minute candles
cannot establish sub-minute reaction timing.

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

## 10. Evidence audit and mismatch adjudication

### 10.1 Assurance boundary and acquisition evidence

SHA-256 proves that bytes have not changed relative to a trusted digest. It
does not prove that the original publisher was truthful or that a person with
write access did not manufacture both an object and a new manifest.

Every new raw object therefore binds to an immutable acquisition record:

```text
source_id
publisher
canonical_url
redirect_chain
request_fingerprint
cursor_or_subscription
fetched_at
response_status
response_date
etag
last_modified
content_type
content_length
byte_length
object_sha256
capture_tool_version
capture_session_id
```

Each sealed capture segment and static acquisition record contributes to a
daily root. The capture service signs the daily root with a recorded
`signing_key_id` and obtains one external trusted timestamp. RFC 3161 is the
initial timestamp protocol. This proves existence no later than the external
timestamp; it does not prove factual correctness.

Existing objects may be anchored now to prevent future undetected rewriting,
but the new anchor cannot be backdated to prove their original acquisition
time. Formal readers accept only verified object-manifest-acquisition chains.

The dataset registry records an `independence_group`. Two delivery surfaces
that share NFL/GSIS or another upstream may corroborate parsing and
publication consistency but do not count as independent observations.

### 10.2 Audit pipeline and verdicts

Every event-to-market candidate passes this ordered pipeline:

```text
raw integrity
-> game-event validity
-> market identity, rules, and orientation
-> temporal eligibility
-> observation and stream validity
-> concurrent-event and microstructure diagnosis
-> economic-surprise classification
```

The public verdict is deliberately small:

```text
PASS
FLAG
REVIEW_REQUIRED
BLOCK
```

Each verdict also carries a scope and reason code. Scopes are `RAW`,
`GAME`, `JOIN`, and `EVENT_RESULT`. Reason codes include:

```text
TAMPER_SUSPECTED
PROVENANCE_INCOMPLETE
PARSER_BUG
SOURCE_CONFLICT
EVIDENCE_INSUFFICIENT
STATE_INVARIANT_FAILED
IDENTITY_AMBIGUOUS
OUTCOME_ORIENTATION_FAILED
RULES_MISMATCH
TIME_INELIGIBLE
STREAM_GAP
WINDOW_CONTAMINATED
OBSERVATION_INELIGIBLE
TRADE_NOISE
WINDOW_SENSITIVE
SURPRISING_DIRECTION
UNEXPECTED_BUT_VALID
```

An upstream quality issue may remain a `FLAG` only while no downstream claim
depends on it. If a requested result requires that evidence, the result is
`BLOCK`.

### 10.3 Game-fact correctness

The structured nflverse source and the official Game Book are parsed
independently. Every derived official fact carries:

```text
raw_sha256
page
bbox_or_line_span
cited_text_sha256
extractor_name
extractor_version
fact_sha256
```

A fact without a deterministic citation round trip cannot enter canonical
replay. Every source row is classified as `MATCHED_PLAY`,
`MATCHED_ANNOTATION`, `EXPLICIT_ADMIN`, or `UNMAPPED`; no row is silently
dropped.

The reducer checks score, period, clock, possession, down, distance, field
position, drive, timeout, no-play, and review ledgers. A review keeps its
provisional fact, review action, and final fact; the canonical reducer consumes
the final fact without pretending that it was a second physical play.

Audit findings distinguish:

- `PARSER_BUG`: fixed source bytes support a different parsed value;
- `SOURCE_CONFLICT`: independently parsed source bytes disagree on the same
  fact;
- `UNEXPECTED_BUT_VALID`: the fixed facts are consistent with a cited final
  ruling or rule exception;
- `EVIDENCE_INSUFFICIENT`: the fact cannot yet be established.

### 10.4 Scoring-team probability-down detector

`score => scoring-team probability must rise` is not a correctness invariant.
The market may have moved before the feed, expected a touchdown but observed a
field goal, priced the cost of returning possession, reacted to a review or
injury, or printed one trade inside a wide spread.

The detector first proves one canonical proposition, such as
`P(DET wins this game)`. It binds the exact game, venue contract, team,
outcome or token, settlement rules, and active interval. It does not infer the
direction from BUY/SELL alone and does not treat a spread, total, or player
prop as a winner market.

For a finalized scoring event:

```text
economic_pre:
  M0 upper bound <= play-start lower bound

available_pre:
  M3 <= G3

economic_post:
  M0 lower bound >= result-finalization upper bound

available_post:
  M3 > G3
```

The detector reports economic and availability order separately. A clean
prospective baseline requires `economic_pre + available_pre`; a clean
event-after observation requires `economic_post + available_post`. Historical
data without `G3` or `M3` uses only economic ordering plus an explicitly
simulated availability profile.

The post observation must precede the next material game event and must not
cross a review, stream gap, suspension, reconnect, or clock epoch. Simulated
availability delay and observation horizons are registered before inspecting
results.

Bid, ask, last trade, and VWAP paths remain separate. Midpoint may be a
diagnostic only and is never executable evidence. A strong quote-confirmed
downward candidate is:

```text
materiality =
  max(2 * tick, 0.5 * max(pre_spread, post_spread))

quote_confirmed_down =
  post_ask < pre_bid - materiality
```

A lower last trade without lower bid and ask is `TRADE_NOISE`, not a confirmed
probability move. A direction that changes across admissible latency profiles
is `WINDOW_SENSITIVE`. A minute candle overlapping the event cannot establish
a sub-minute event response.

A valid downward candidate becomes `REVIEW_REQUIRED`; it never changes raw
score, time, or price and never blocks the whole game by itself. Integrity,
identity, rules, state, stream, or time failures block the affected scope.

### 10.5 Human and LLM review boundary

The review bundle contains:

- candidate ID, verdict, reason codes, configuration hashes, and raw refs;
- pre-state, event, post-state, revision chain, and official citations;
- exact contract, target team, rules hash, and orientation proof;
- all time intervals, clock classes, uncertainties, and next material event;
- bid, ask, trades, VWAP, depth, spread, tick, volume, and quote age;
- a PASS/FAIL/UNKNOWN counter-evidence matrix.

An LLM may propose cited extraction or matching candidates on the slow path.
It cannot issue `PASS`, alter a fact, infer missing historical timestamps,
resolve a source conflict, or repair market direction.

A human reviewer appends a signed resolution record. Allowed decisions are
`CONFIRMED_INPUT_ERROR`, `CONFIRMED_MAPPING_ERROR`,
`CONFIRMED_TIME_ERROR`, `VALID_MARKET_SURPRISE`,
`MICROSTRUCTURE_NOISE`, `PLAUSIBLE_STATE_RESPONSE`, and `UNRESOLVED`.
The reviewer cannot edit raw or overwrite an earlier artifact. A correction
creates a new derived version linked to the original hashes; `UNRESOLVED`
remains blocked from formal results.

## 11. Artifacts

The design introduces these derived artifacts:

```text
contracts/temporal-evidence/v0.schema.yaml
contracts/capture-timing/v0.schema.yaml
contracts/acquisition-evidence/v0.schema.yaml
contracts/audit-verdict/v0.schema.yaml
contracts/market-reaction-candidate/v0.schema.yaml
contracts/game-replay/v1.schema.yaml

artifacts/time/nfl_2025_14_dal_det_time_audit_v0.json
artifacts/audit/daily/<date>-capture-root-v0.json
artifacts/game-state/nfl/nfl_2025_14_dal_det_events_v1.jsonl
artifacts/game-state/nfl/nfl_2025_14_dal_det_state_trace_v1.jsonl
artifacts/game-state/nfl/nfl_2025_14_dal_det_state_replay_v1.json
artifacts/game-state/nfl/nfl_2025_14_dal_det_gamebook_audit_v0.json
artifacts/audit/nfl/nfl_2025_14_dal_det_event_market_audit_v0.json
```

Operational timing sidecars remain outside canonical event identity and bind
to immutable raw records by session, ordinal, and payload hash.

Existing v0 state artifacts remain immutable evidence. New consumers migrate
atomically to the v1 replay contract; no dual-read compatibility layer is
introduced.

## 12. Delivery order and ownership

### Stage 1: temporal and audit contracts

Owner: A + C + H

- freeze timestamp roles, intervals, clock domains, uncertainty math, and
  percentile estimator;
- publish temporal-evidence, capture-timing, acquisition-evidence,
  audit-verdict, and market-reaction-candidate contracts;
- freeze receive-boundary names, clock-quality classes, error budgets, verdict
  scopes, reason codes, and fail-closed eligibility rules;
- add dataset `independence_group` and source-publisher evidence.

Gate: the same fixtures validate game and market timestamps without
source-specific private fields; no formal reader can bypass verified
object-manifest-acquisition evidence.

### Stage 2: historical NFL state replay

Owner: D2 + C + H

- upgrade the DAL-DET event and replay artifacts to v1;
- add play completion, finalization, revision, and time-evidence fields;
- complete official Game Book event mapping;
- add deterministic citation round trips and row classifications;
- exercise review reversal, upheld challenge, no-play, score, possession, and
  clock exceptions;
- retain current hashes and artifacts as immutable prior evidence.

Gate: full state-replay and official-reference checks pass, with deterministic
Level 1 and Level 2 hashes and no unresolved state-changing row.

### Stage 3: prospective timing instrumentation

Owner: C + H

- stamp local receive before parsing;
- label the application, kernel-software, or NIC-hardware receive boundary;
- record monotonic, UTC, clock-sync, connection, and gap evidence;
- stamp raw-append, parse, normalize, and state/book-ready boundaries;
- classify each epoch and enforce its clock-error and sample-age budget;
- sign and externally timestamp each daily capture root;
- validate current Polymarket recorder timing without reclassifying historical
  records.

Gate: clock-step, stale synchronization evidence, reconnect, duplicate,
sequence-gap, wrong receive-boundary, and unsynced-clock tests fail closed.

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
- generate the first descriptive probability-trend report;
- generate event-level audit bundles and scoring-direction candidates;
- require review resolution for surprising but otherwise valid downward
  movements.

Gate: no alpha, causality, executable-price, or latency claim exceeds the
available timestamp and quote evidence; an economic surprise alone never
rewrites raw data or blocks the full game.

## 13. Acceptance tests

Required tests include:

- timestamp precision expands to the correct half-open interval;
- application callback, kernel software, and NIC hardware timestamps cannot be
  relabelled as one another;
- absent source/publish/receive time remains absent;
- wall-clock steps split timing epochs;
- monotonic durations remain valid across wall-clock offset changes;
- monotonic timestamps from different hosts or boots cannot be subtracted;
- stale or `UNSYNCED` clock evidence blocks cross-clock timing;
- local clock health cannot promote a source timestamp whose external clock
  error or application point is unknown;
- negative cross-clock differences inside uncertainty become unresolved and
  those outside uncertainty fail;
- source precision and clock error enter interval subtraction exactly once;
- a historical source-only record cannot be upgraded by a scenario delay;
- sidecar session, ordinal, and payload-hash mismatch fails;
- historical `fetched_at` cannot enter an availability or latency metric;
- object or manifest tampering fails, and a self-consistent replacement
  object without the signed external anchor remains ineligible;
- an external anchor created today cannot prove an earlier acquisition time;
- review reversal produces separate provisional and final revisions;
- shuffled input cannot alter canonical replay output;
- missing, duplicate, or conflicting source order fails or receives an
  explicit exclusion reason;
- a sequence gap remains invalid until a verified recovery snapshot and
  continuous new sequence open a new segment;
- every Game Book fact round-trips to fixed bytes and a page/span hash;
- an unmatched state-changing row blocks the game replay;
- all DAL-DET state-changing events reconcile to official evidence;
- two replay runs produce identical semantic output and canonical hash;
- interval-overlap observations cannot be classified as definitely post;
- a pre-event market update received after `G3` cannot become the available
  pre-event baseline;
- an `M0` before game finalization with `M3` after `G3` cannot become a clean
  event-after reaction;
- candles cannot produce sub-minute reaction timing;
- unproven outcome orientation blocks probability normalization;
- unregistered availability-profile changes invalidate the result;
- scoring-team probability down produces `REVIEW_REQUIRED`, not an automatic
  data correction or full-game block;
- a lower last trade with unchanged bid and ask is classified as trade noise,
  not a quote-confirmed move;
- an initial 100-sample profile cannot promote P99 as a stable percentile;
- reviewer resolution is append-only and cannot overwrite raw or prior
  findings.

## 14. Completion definition

This phase is complete only when:

1. one time contract governs both G and M;
2. DAL-DET state replay passes its deterministic and official-reference gates;
3. historical availability limitations are explicit and machine-enforced;
4. prospective capture can measure same-host G2/G3/M2/M3 timing;
5. every timing result declares its HFT sampling boundary, clock class,
   uncertainty, epoch, and registered error budget;
6. P95 is source-specific and adjustable only through a versioned profile;
7. capture evidence has a signed, externally timestamped daily root;
8. every state-changing Game Book fact has a deterministic citation round
   trip;
9. economic surprises create immutable review bundles without changing raw
   facts;
10. J remains blocked until both upstream streams and the requested result
    scope independently pass.

## 15. Primary technical references

- [Linux kernel network timestamping](https://docs.kernel.org/networking/timestamping.html)
  for software, kernel, and NIC hardware receive boundaries;
- [Linux PTP hardware clock infrastructure](https://docs.kernel.org/driver-api/ptp.html)
  and [linuxptp `phc2sys`](https://www.linuxptp.org/documentation/phc2sys/)
  for PHC-to-system-clock synchronization and evidence;
- [RFC 3161 Time-Stamp Protocol](https://www.rfc-editor.org/rfc/rfc3161.html)
  for external proof-of-existence of signed capture roots.
