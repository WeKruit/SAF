# Game-State Engine Design

Date: 2026-07-23  
Status: approved by the active program objective  
Program status: `CONDITIONAL_GO`

## Objective

Validate, without trading claims, that each supported sport can:

1. represent the point-in-time game state without future information;
2. deterministically apply one newly observed event to obtain the next state;
3. emit a calibrated distribution for the next registered transition target;
4. report reducer, feature, inference, and end-to-end latency separately; and
5. later join the model output to the matching prediction-market state as of the
   same observation time.

The design preserves the Charter architecture. It adds a domain layer between
the existing event envelope and `model-output/v1`; it does not create a service
or a second control plane.

## Research decision

The first implementation reproduces mature, interpretable work:

- NFL: nflverse play-by-play state, nflfastR/Lock-Nettleton style win
  probability, and a next-drive classifier.
- Soccer: StatsBomb event state, SPADL/VAEP-style action context, with
  Dixon-Coles retained only as a pre-game prior.
- MLB: Retrosheet parsed through Chadwick semantics, a deterministic
  base/out state, and the classical 24-state Markov baseline.
- NBA: possession-level interface and fixtures only until the NBA data-rights
  gate is green.
- F1: lap-level interface and fixtures only until data rights and timestamp
  semantics are green.

Tracking-data EPV, sequence foundation models, RL, and one universal
cross-sport feature vector are excluded from the first implementation.

Primary references:

- nflverse data and nflfastR models:
  <https://github.com/nflverse/nflverse-data> and
  <https://opensourcefootball.com/posts/2020-09-28-nflfastr-ep-wp-and-cp-models/>
- StatsBomb Open Data and VAEP:
  <https://github.com/statsbomb/open-data> and
  <https://arxiv.org/abs/1802.07127>
- Retrosheet, Chadwick, and the 24-state Markov model:
  <https://www.retrosheet.org/game.htm>,
  <https://chadwick.readthedocs.io/en/stable/>, and
  <https://doi.org/10.1287/opre.45.1.14>
- NBA terms and FastF1 data reference:
  <https://www.nba.com/termsofuse> and
  <https://docs.fastf1.dev/data_reference/index.html>

## Boundary and data flow

```text
native event
  -> EventEnvelopeV0(normalized_observation)
  -> sport adapter
  -> immutable SportEvent
  -> deterministic SportReducer(current_state, event)
  -> immutable next SportState
  -> sport feature extractor
  -> registered transition predictor
  -> ModelOutputV1(state_event_id = consumed envelope event_id)
```

The deterministic reducer and probabilistic predictor are separate:

```text
S[t+1] = reduce(S[t], observed_event[t+1])
P(next_transition | S[t+1]) = predict(features(S[t+1]))
```

An observed event may contain source-reported post-event fields. Such fields
must be available in the event payload at the declared cutoff and must never be
read from a later row silently. Offline adapters must label this distinction.

## Common interface

All states expose:

- `sport`
- canonical `game_id`
- monotonically increasing `sequence`
- `terminal`

All events expose:

- `sport`
- the same canonical `game_id`
- the next exact `sequence`
- the real `evt_<sha256>` event-envelope identifier

The common layer:

- validates identity, order, terminal state, and reducer output;
- creates canonical previous/event/next hashes;
- rejects binary floats in hash material;
- returns a trace containing the immutable next state;
- benchmarks reducer, feature extraction, prediction, and end-to-end stages.

Each sport owns an immutable state and event schema. There is no union schema
whose unused fields become nullable.

## Sport state minimums

### NFL

Period and clocks, possession and defense, home/away scores, down, distance,
field position, goal-to-go, timeouts, drive/play identifiers, play phase, and
terminal state. Offline reconstruction may compare a reduced state with the
next observed pre-play snapshot, but EPA/WPA, final drive result, and postgame
fields are forbidden predictor features.

### Soccer

Period and clock, score, home/away teams, possession, play pattern, ball
location, in-play status, last action, cards, substitutions, active players,
and terminal state. Coordinates use scaled integers, never binary floats.

### MLB

Inning/half, outs, base runners, score, batting/fielding teams, batter, pitcher,
count, lineup slot, and terminal state. The raw Retrosheet grammar remains the
responsibility of Chadwick; the SAF reducer consumes a parsed, explicit play
observation.

## Correctness validation

For every reducer:

- the same state and event produce the same next state and hashes;
- cross-game/cross-sport events fail closed;
- duplicate, skipped, or reversed sequence numbers fail closed;
- clocks and sport-specific invariants fail closed;
- terminal states reject new events;
- representative scoring, possession, period/inning, card/substitution, and
  timeout transitions have explicit tests.

Real-data validation is game-grouped and chronological. It reports exact state
field agreement, mismatch categories, unsupported-event rate, missing-field
rate, and state-hash replay agreement. A runner cannot convert a fixture result
into empirical evidence.

## Latency standard

Batch size is one. Every report fixes code/data/config hashes, Python and
library versions, CPU/OS, warm-up count, repetitions, event count, and clock.
It reports p50/p95/p99/max and throughput for:

1. reducer;
2. feature extraction;
3. model inference;
4. complete state-event-to-output path.

Full repository test duration is not model latency.

## Prediction-market alignment gate

No output is called alpha until all of the following exist:

- canonical game-to-condition mapping;
- sport event source time and local receive time;
- model output cutoff and market quote as-of time;
- executable bid/ask rather than midpoint;
- market age, spread, depth, pause, and rule snapshot;
- paired model-minus-market calibration and lead/lag confidence intervals.

Before this gate the only permitted label is `predictive_disagreement`.

## NO-GO

All Charter NO-GO items remain active. This design does not authorize real
money, maker execution, queue-fill claims, multi-venue live arbitrage, live
copy trading, LLM hot path, RL, or an unregistered empirical run.
