# Taker Simulator Specification v0

- **Owner:** Team F1
- **Version:** v0
- **Experiment:** `X-07`
- **Due gate:** `2026-08-05_W2_review`; O-002 remains gated by `Team_I_compliance_green`
- **Status:** `PRELIMINARY_ONLY`

The v0 simulator is taker-only. It cannot place orders and has no venue credentials. It emits one canonical `TcaRecordV0` only from an immutable book stream and the append-only `VenueRuleStore`. It does not accept a caller-supplied snapshot.

## Buy execution

1. Resolve the latest valid, non-stale rule for the order venue, market, and condition strictly as of order creation. Missing, invalid, future-only, or stale rules fail closed.
2. Require every book to match the order venue and condition and to carry `local_receive_at` plus a verified `content_sha256` over its complete v0 content.
3. Compute eligibility time as order creation + snapshot `seconds_delay` + explicit own-system delay, then skip earlier and suspended books.
4. Consume asks in ascending price order until the exact fixed-point quantity is filled. Apply the snapshot fee formula to each consumed depth level and sum the exact fees; applying the nonlinear formula once to aggregate VWAP is forbidden.
5. At each explicit markout horizon, consume bid depth in descending order for the same quantity. Resolve the rule again as of the actual exit book, apply exit fees to each consumed depth level, and report executable bid VWAP, gross markout, exit fee, and net markout after both entry and exit fees.
6. Store entry and exit rule hashes, book content hashes, fixed-point costs, delays, and levels consumed in `TcaRecordV0`. The runtime model and `contracts/tca/v0.schema.yaml` have identical fields and must survive canonical JSON validation, hashing, and replay without change.

No midpoint, last trade, future depth, constant fee, or constant venue delay is accepted. Insufficient depth is a hard error; a separate top-of-book calculation may be reported only as an explicitly optimistic bound and never as a depth-VWAP fill.

## Point-in-time rule contract

The order fixes venue, market, and condition. `VenueRuleStore` performs the strict as-of selection from the same canonical store used by replay; timestamp checks on an arbitrary object are not a substitute. Each entry or exit TCA leg stores the canonical SHA-256 of its validated rule snapshot. v0 implements only `C × rate × (p × (1-p))^exponent` at each actual fill price and rejects a non-integer exponent rather than applying context-sensitive rounding.

## Current promotion boundary

Every output is labeled `PRELIMINARY`. A request for formal output is rejected even when it supplies a snapshot because X-07 `formal_result` is currently unauthorized and `venue_rule_snapshot_replay_version` is unresolved. Formal promotion requires the registered locks, snapshot replay consumption, code/data preregistration, and Team H result validation. This specification is not a go/no-go input.
