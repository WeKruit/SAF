# Market Join and Reaction Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a fail-closed NFL event × market audit that detects surprising price direction without confusing it with data corruption or execution evidence.

**Architecture:** Verify venue identity, outcome orientation, rules, raw lineage, observation type, and temporal eligibility before calculating any movement. The detector keeps bid, ask, trades, and VWAP separate; it emits immutable review bundles with PASS/FLAG/REVIEW_REQUIRED/BLOCK verdicts.

**Tech Stack:** Python 3.12, existing market alignment and NFL mapping modules, Pydantic v2, pytest.

---

## File structure

- Modify: `src/prediction_market/sports/nfl_market_observation.py` — read verified manifests and prove exact condition/token/ticker/team direction.
- Modify: `src/prediction_market/sports/market_alignment.py` — consume temporal eligibility and observation-quality gates.
- Create: `src/prediction_market/sports/market_reaction_audit.py` — movement measures, anomaly classification, and review bundle writer.
- Test: `tests/sports/test_nfl_market_observation.py`, `tests/sports/test_market_alignment.py`, `tests/sports/test_market_reaction_audit.py`.

### Task 1: Strengthen market identity and orientation

**Files:**
- Modify: `src/prediction_market/sports/nfl_market_observation.py`
- Test: `tests/sports/test_nfl_market_observation.py`

- [ ] Test that Polymarket requires `condition_id + token_id + outcome label`, and Kalshi requires ticker, `yes_sub_title`, target team, and rules evidence. Reject a hash-shaped manifest reference that cannot be read and verified.
- [ ] Test that a BUY/SELL side never changes token probability orientation by itself and that a spread/total/player market cannot satisfy a winner-market request.
- [ ] Replace caller-provided completeness claims with verified manifest reads and exact native mapping evidence.
- [ ] Run: `uv run pytest tests/sports/test_nfl_market_observation.py -q`. Expected: PASS.
- [ ] Commit: `git commit -m "feat: require verified NFL market orientation"`.

### Task 2: Enforce time, stream, and observation gates

**Files:**
- Modify: `src/prediction_market/sports/market_alignment.py`
- Test: `tests/sports/test_market_alignment.py`

- [ ] Test that missing M3/G3, a clock epoch mismatch, an ambiguous source timestamp, a reconnect gap, a crossed quote, a stale quote, or an overlapping candle blocks the affected event result.
- [ ] Test that a source-time-pre update received after game-ready time cannot become the pre-event baseline, and an update before finalization cannot become a clean event-after observation.
- [ ] Require the temporal engine's dual-order verdict and the capture-side sequence recovery status before formal alignment is eligible.
- [ ] Run: `uv run pytest tests/sports/test_market_alignment.py tests/test_temporal.py -q`. Expected: PASS.
- [ ] Commit: `git commit -m "feat: gate market alignment on temporal evidence"`.

### Task 3: Build direction detector and immutable review bundle

**Files:**
- Create: `src/prediction_market/sports/market_reaction_audit.py`
- Test: `tests/sports/test_market_reaction_audit.py`

- [ ] Test a quote-confirmed downward movement, trade-only noise, window-sensitive result, cross-venue disagreement, next-event contamination, and an otherwise valid market surprise.
- [ ] Implement the exact materiality predicate:
  `max(2 * tick, 0.5 * max(pre_spread, post_spread))`.
  Mark quote-confirmed down only when `post_ask < pre_bid - materiality`.
- [ ] Keep `delta_bid`, `delta_ask`, `delta_last_trade`, and `delta_vwap` separate. Midpoint is diagnostic only.
- [ ] Write a canonical review bundle containing raw refs, state revision chain, rules/orientation proof, all intervals and clock classes, market microstructure fields, counter-evidence matrix, and append-only reviewer decision.
- [ ] A score followed by a downward move must be `REVIEW_REQUIRED` unless an integrity/identity/rules/time gate already produces `BLOCK`; it must never modify raw data.
- [ ] Run: `uv run pytest tests/sports/test_market_reaction_audit.py tests/sports/test_market_alignment.py -q`. Expected: PASS.
- [ ] Commit: `git commit -m "feat: audit NFL market reaction direction"`.

## Plan self-review

- Coverage: exact venue mapping, source/availability timing, quote versus trade semantics, contamination, anomaly diagnosis, and append-only human adjudication.
- Boundary: output remains descriptive and cannot claim alpha, execution, causality, or fair value.
