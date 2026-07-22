# X-09 Deterministic Vertical Slice Harness v0

- **Owner:** Team B+A+H
- **Experiment:** `X-09`
- **Artifact status:** `HARNESS_PASS`
- **Experiment status:** `EXPERIMENT_BLOCKED`
- **Authority:** `contracts/deterministic-replay/v0.md` and the registered X-09 card

## Purpose

This harness exercises one deterministic, simulated-only path:

`normalized data -> score+5s buy signal -> risk check -> simulated order -> taker fill -> P&L -> canonical event log`

It is engineering evidence that the path can satisfy replay Levels 1 and 2 for a frozen fixture. It is not an X-09 formal result. The registered X-09 input, rule, risk/fill, serializer/seed/lockfile, and H-approval locks are unresolved, and X-01 is not complete.

## Boundaries

- The input is a frozen tuple of validated `EventEnvelopeV0` observations plus one validated `VenueRuleSnapshotV0` and immutable fixed-point configuration.
- All prices, quantities, limits, fees, and P&L values use `FixedPointV0`/`Decimal`; binary floats are invalid.
- No venue credentials, network calls, wall clock, real orders, or random runtime state participate.
- The X-07 taker result types are not reused because they are deliberately scoped to `X-07/PRELIMINARY`. X-09 emits its own `experiment_id: X-09` simulated envelopes while retaining the shared event and rule-snapshot contracts.

## Deterministic flow

1. Validate and internally sort the frozen input envelopes by the complete replay total-order key.
2. Select the score observation and create exactly one buy signal at score receive time plus five seconds.
3. Apply the frozen maximum-quantity risk limit. A rejected quantity fails closed and emits no order.
4. Create one simulated taker-buy order at the signal timestamp.
5. As-of validate the per-condition venue rule at order time. Eligibility is order time plus the snapshot venue delay.
6. Select the first non-suspended book at or after eligibility and consume asks in ascending-price order for exact depth VWAP.
7. Select the first non-suspended book at or after the frozen P&L horizon and consume bids in descending-price order for executable proceeds.
8. Emit exact fee, total cost, proceeds, and P&L, then sort the complete input/output stream and serialize each validated envelope as canonical JSON.

Insufficient depth, missing score/book/rule data, stale or mismatched rules, non-UTC time, invalid fixed point, ambiguous input, or a semantic/hash mismatch is a hard error.

## Lineage and rule evidence

- `signal` directly names the score event as its parent.
- `simulated_order` directly names both the score trigger and signal event as parents.
- `simulated_fill` names the order and execution-book events as parents.
- `simulated_pnl` names the fill and exit-book events as parents.
- Every `simulated_order`, `simulated_fill`, and `simulated_pnl` carries the canonical SHA-256 of the complete validated rule snapshot as `rule_snapshot_ref`.

## Replay evidence

Level 1 compares the full ordered event identity/type sequence, canonical order payloads, canonical fill payloads, canonical P&L payloads, and terminal P&L. Level 2 uses the contract-defined domain/count/event-ID framing and SHA-256. The canonical log is newline-delimited canonical JSON in the same total order; byte identity is recorded but is not a Level-3 claim.

The harness runs the same frozen input twice. Only an exact Level-1 match and an exact Level-2 digest match may return:

- `harness_status: HARNESS_PASS`
- `experiment_status: EXPERIMENT_BLOCKED`

## Formal entry gate

The formal entry reads the repository experiment registry on every invocation and fails closed unless all of the following are true:

1. X-09 formal scope is authorized and every required registration lock is resolved.
2. X-01 has completed validated dependency evidence.
3. X-09 formal code and data hashes were preregistered and match this runtime and exact frozen fixture.
4. Two executions match at Levels 1 and 2.

The current registry cannot satisfy these conditions. This artifact therefore contains no `FORMAL`, X-09 pass, promotion, or backtest-unfreeze claim.

## Verification

Focused verification is `uv run pytest tests/replay -q` (11 tests). Repository diff hygiene is checked with `git diff --check` before commit.
