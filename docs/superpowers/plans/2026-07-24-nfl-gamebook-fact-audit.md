# NFL Game Book Fact Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the DAL–DET game-state replay against immutable Game Book evidence without allowing an LLM or a human to rewrite source facts.

**Architecture:** Store the official PDF as a verified static object, extract page/span-cited facts into a separate immutable artifact, and match nflverse rows deterministically. The reducer consumes final review facts only; every mismatch remains an auditable finding.

**Tech Stack:** Python 3.12, PyArrow, Pydantic v2, existing static store, deterministic PDF text extractor, pytest.

---

## File structure

- Create: `src/prediction_market/sports/nfl_gamebook_audit.py` — cited fact model, matcher, invariants, and audit report.
- Modify: `src/prediction_market/sports/nfl_game_replay.py` — require a verified Game Book audit before publishing replay v1.
- Modify: `src/prediction_market/sports/nfl_game_state.py` — expose final-review provenance where missing.
- Test: `tests/sports/test_nfl_gamebook_audit.py`, `tests/sports/test_nfl_game_replay.py`.
- Add fixture: `tests/fixtures/nfl/gamebook_dal_det_facts_v0.json` — bounded hand-verified facts with PDF page/span/hash citations.

### Task 1: Cited Game Book fact contract and fixture

**Files:**
- Create: `src/prediction_market/sports/nfl_gamebook_audit.py`
- Create: `tests/fixtures/nfl/gamebook_dal_det_facts_v0.json`
- Test: `tests/sports/test_nfl_gamebook_audit.py`

- [ ] Test rejection of a fact with no source object hash, no page/span, a mismatched cited-text hash, or a provisional score presented as final.
- [ ] Run: `uv run pytest tests/sports/test_nfl_gamebook_audit.py -q`. Expected: FAIL because the audit module is absent.
- [ ] Implement `GameBookCitationV0`, `GameBookFactV0`, and `verify_citation_round_trip`. A fact must reference `raw_sha256 + page + span + cited_text_sha256 + extractor_version`.
- [ ] Include fixtures for the reversed safety, overturned first down, upheld challenge/timeout, and no-play penalty cases.
- [ ] Run: `uv run pytest tests/sports/test_nfl_gamebook_audit.py -q`. Expected: PASS.
- [ ] Commit: `git commit -m "feat: add cited NFL Game Book facts"`.

### Task 2: Deterministic row classification and state invariants

**Files:**
- Modify: `src/prediction_market/sports/nfl_gamebook_audit.py`
- Test: `tests/sports/test_nfl_gamebook_audit.py`

- [ ] Test that every nflverse row is one of `MATCHED_PLAY`, `MATCHED_ANNOTATION`, `EXPLICIT_ADMIN`, or `UNMAPPED`, and that an unmatched state-changing row blocks the game.
- [ ] Test score ledger closure, review finality, clock exception citations, possession/drive closure, and no-play spot handling.
- [ ] Implement order-preserving matching by game identity, period, clock, offense, down-distance, yardline, and normalized description; never use replay output to repair a match.
- [ ] Emit `PARSER_BUG`, `SOURCE_CONFLICT`, `UNEXPECTED_BUT_VALID`, or `EVIDENCE_INSUFFICIENT` explicitly.
- [ ] Run: `uv run pytest tests/sports/test_nfl_gamebook_audit.py -q`. Expected: PASS.
- [ ] Commit: `git commit -m "feat: audit NFL replay facts against Game Book"`.

### Task 3: Bind Game Book audit to replay v1

**Files:**
- Modify: `src/prediction_market/sports/nfl_game_replay.py`
- Test: `tests/sports/test_nfl_game_replay.py`

- [ ] Test that replay v1 refuses a missing, unverified, or unresolved Game Book audit while preserving all v0 artifacts unchanged.
- [ ] Add an audit summary hash, row-coverage counts, and open-finding counts to `NFLGameReplay.summary()`.
- [ ] Require all state-changing rows to have resolved final facts before formal replay v1 is written.
- [ ] Run: `uv run pytest tests/sports/test_nfl_game_replay.py tests/sports/test_nfl_gamebook_audit.py -q`. Expected: PASS.
- [ ] Commit: `git commit -m "feat: gate NFL replay on Game Book audit"`.

## Plan self-review

- Coverage: immutable source citation, full row accountability, special NFL review cases, invariants, and no-LLM/no-human overwrite boundary.
- Boundary: this plan does not bind market data or infer event-time availability; it exports only accepted game facts.
