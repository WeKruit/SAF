# Temporal Evidence and Capture Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make prospective captures carry machine-verifiable timestamp semantics, clock quality, interval eligibility, and acquisition provenance.

**Architecture:** New v0 contracts and pure temporal functions sit outside immutable raw bytes. Recorder timing is an append-only sidecar keyed by session, ordinal, and payload hash. Daily roots bind verified capture manifests to an externally supplied RFC 3161 token.

**Tech Stack:** Python 3.12, Pydantic v2, cryptography, pytest.

---

## File structure

- Modify: `src/prediction_market/contracts.py` — temporal, capture-timing, acquisition, and audit models.
- Create: `src/prediction_market/temporal.py` — interval arithmetic and dual-order G/M eligibility.
- Create: `src/prediction_market/capture_timing.py` — sandwich sampling and sidecar rows.
- Create: `src/prediction_market/acquisition.py` — daily roots and anchor verification.
- Modify: `src/prediction_market/recording.py`, `src/prediction_market/adapters/base.py` — timing boundaries and recovery state.
- Test: `tests/contracts/test_temporal_evidence_contract.py`, `tests/test_temporal.py`, `tests/test_capture_timing.py`, `tests/test_acquisition.py`, `tests/test_recording.py`.

### Task 1: Add normative contracts

**Files:**
- Modify: `src/prediction_market/contracts.py`
- Test: `tests/contracts/test_temporal_evidence_contract.py`

- [x] Write tests rejecting an empty half-open interval, a UTC-bounded record without `clock_error_ns`, and a signed daily root whose token hash differs from its stored token.
- [x] Run: `uv run pytest tests/contracts/test_temporal_evidence_contract.py -q`. Expected: FAIL because the v0 models are absent.
- [x] Add `TemporalEvidenceV0`, `CaptureTimingV0`, `AcquisitionEvidenceV0`, `DailyCaptureRootV0`, and `AuditVerdictV0`. Each model must reject unknown status literals, non-positive interval width, and a missing numeric error bound for `UTC_BOUNDED_*`.
- [x] Run: `uv run pytest tests/contracts/test_temporal_evidence_contract.py tests/contracts/test_contracts.py -q`. Expected: PASS.
- [x] Commit: `git commit -m "feat: add temporal evidence contracts"`.

### Task 2: Add dual-order temporal gates

**Files:**
- Create: `src/prediction_market/temporal.py`
- Test: `tests/test_temporal.py`

- [ ] Test the exact predicates: economic pre requires `M0.upper <= play_start.lower`; economic post requires `M0.lower >= finalization.upper`; available pre requires `M3 <= G3`; available post requires `M3 > G3`.
- [ ] Test that source precision and clock error enter `B - A = [B.lower - A.upper, B.upper - A.lower)` exactly once, and different host/boot/clock/epoch monotonic values cannot be subtracted.
- [ ] Implement `UtcInterval`, `LocalInstant`, `subtract_intervals`, `same_host_elapsed`, and `classify_event_market_order`; return explicit blocked/unresolved states instead of clamping negative values.
- [ ] Run: `uv run pytest tests/test_temporal.py -q`. Expected: PASS.
- [ ] Commit: `git commit -m "feat: add fail-closed temporal join gates"`.

### Task 3: Persist capture sidecars and sequence recovery barriers

**Files:**
- Create: `src/prediction_market/capture_timing.py`
- Modify: `src/prediction_market/recording.py`
- Modify: `src/prediction_market/adapters/base.py`
- Test: `tests/test_capture_timing.py`, `tests/test_recording.py`

- [ ] Test a monotonic-UTC-monotonic sandwich, an expired clock sample, a callback delay after socket receipt, a sequence gap, and a reconnect that lacks a recovery snapshot.
- [ ] Sample callback timing immediately after `websocket.recv()` returns and before parsing. Persist the timing row only after raw append succeeds; include raw-append, parse, normalize, and state-ready monotonic boundaries.
- [ ] Extend `SequenceTracker` to retain sequence domain, epoch, gap range, and recovery snapshot ID. A gap must block incremental validity until a verified snapshot opens a new segment.
- [ ] Run: `uv run pytest tests/test_capture_timing.py tests/test_recording.py tests/adapters/test_base.py -q`. Expected: PASS.
- [ ] Commit: `git commit -m "feat: record timing and sequence recovery evidence"`.

### Task 4: Build daily roots and require anchors for formal reads

**Files:**
- Create: `src/prediction_market/acquisition.py`
- Test: `tests/test_acquisition.py`

- [ ] Test deterministic daily-root ordering, duplicate-manifest rejection, signer verification, missing RFC 3161 token rejection under formal mode, and rejection of an anchor created after the asserted acquisition time.
- [ ] Implement a root over unique verified manifest hashes, signature verification, signer-key ID binding, and timestamp-token digest binding. Treat the TSA token as externally supplied bytes; do not invent historical anchors.
- [ ] Run: `uv run pytest tests/test_acquisition.py -q`. Expected: PASS.
- [ ] Commit: `git commit -m "feat: add anchored daily capture roots"`.

## Plan self-review

- Coverage: contracts, HFT sampling boundary, clock-error interval math, sequence recovery, immutable timing sidecars, and acquisition anchors are implemented here.
- Boundaries: Game Book fact validation and market-direction review live in their own plans; this plan exports the contracts they consume.
- No compatibility reader is retained: formal consumers must opt into verified temporal/acquisition artifacts.
