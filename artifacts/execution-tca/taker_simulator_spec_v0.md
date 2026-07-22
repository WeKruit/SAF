# Taker Simulator Specification v0

- **Owner:** Team F1
- **Version:** v0
- **Experiment:** `X-07`
- **Due gate:** `2026-08-05_W2_review`; O-002 remains gated by `Team_I_compliance_green`
- **Status:** `PRELIMINARY_ONLY`

The v0 simulator is taker-only. It cannot place orders and has no venue credentials. It emits a TCA record only from an immutable book stream and an exact point-in-time `venue_rule_snapshot`.

## Buy execution

1. Compute eligibility time as order creation + snapshot `seconds_delay` + explicit own-system delay.
2. Skip books before eligibility and all suspended books.
3. Consume asks in ascending price order until the exact fixed-point quantity is filled.
4. Report depth VWAP, gross cost, the snapshot-derived fee, total cost, and levels consumed.
5. At each explicitly supplied markout horizon, consume bid depth in descending order for the same quantity and report executable bid VWAP and gross per-unit markout.

No midpoint, last trade, future depth, constant fee, or constant venue delay is accepted. Insufficient depth is a hard error; a separate top-of-book calculation may be reported only as an explicitly optimistic bound and never as a depth-VWAP fill.

## Point-in-time rule contract

The snapshot condition must equal the order condition. Both `fetched_at` and `effective_from` must be no later than order creation. The TCA record stores the canonical SHA-256 of the complete validated snapshot. v0 implements only the explicitly selected `C × rate × (p × (1-p))^exponent` fee formula and rejects a non-integer exponent rather than applying context-sensitive rounding.

## Current promotion boundary

Every output is labeled `PRELIMINARY`. A request for formal output is rejected even when it supplies a snapshot because X-07 `formal_result` is currently unauthorized and `venue_rule_snapshot_replay_version` is unresolved. Formal promotion requires the registered locks, snapshot replay consumption, code/data preregistration, and Team H result validation. This specification is not a go/no-go input.
