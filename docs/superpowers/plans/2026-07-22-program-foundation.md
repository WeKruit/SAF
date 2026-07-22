# Program Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the approved first-round repository, contracts, registries, recorders, audit/replay tools, research POCs, labeling/execution specifications, strategy audit, and compliance artifacts for Teams A–I.

**Architecture:** A Python 3.12 contract-first modular monolith separates native venue adapters from the unified control plane through typed in-process ports. Immutable raw records, canonical fixed-point events, versioned venue rules, and append-only experiment metadata provide deterministic replay and auditability.

**Tech Stack:** Python 3.12, uv, Pydantic v2, PyYAML, httpx, websockets, cryptography, zstandard, DuckDB, PyArrow, pandas, scikit-learn, pytest, pytest-asyncio.

---

## Global execution rules

- The three Charter files are the only program-level sources of truth.
- Every worker owns only the files listed in its task and must not revert other workers' changes.
- Production code follows test-first RED/GREEN/REFACTOR.
- Documentation-only artifacts are verified by schema/link/registry tests rather than untestable prose assertions.
- No real-money execution, maker model, multi-venue live arbitrage, live copy trading, LLM hot path, RL, or service decomposition may be added.
- Every task ends with its focused tests, the full test suite, governance validation, and an atomic commit.

## Task 1: Repository and governance baseline — Team A/H

**Files:**

- Create: `pyproject.toml`
- Create: `src/prediction_market/__init__.py`
- Create: `src/prediction_market/governance.py`
- Create: `tools/validate_governance.py`
- Create: `tests/test_governance.py`
- Create: `README.md`

- [ ] **Step 1: Write failing governance tests**

```python
def test_program_sources_match_manifest(program_root):
    report = validate_program(program_root)
    assert report.source_hashes_valid is True

def test_catalog_has_one_primary_owner_per_item(program_root):
    report = validate_program(program_root)
    assert report.catalog_rows == 87
    assert report.assignment_rows == 150
    assert report.items_without_one_primary == ()
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_governance.py -q`  
Expected: import failure for `prediction_market.governance`.

- [ ] **Step 3: Implement minimal governance validator and project metadata**

`validate_program(root)` must verify manifest hashes, exact source filenames, unique catalog IDs, known assignment IDs, and exactly one primary owner per catalog item. `tools/validate_governance.py` exits nonzero and prints violations when any invariant fails.

- [ ] **Step 4: Verify GREEN and full baseline**

Run: `uv run pytest tests/test_governance.py -q`  
Expected: all tests pass.  
Run: `uv run python tools/validate_governance.py`  
Expected: `87 catalog items; 150 assignments; governance valid`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md src/prediction_market tools/validate_governance.py tests/test_governance.py uv.lock
git commit -m "build: establish governed program repository"
```

## Task 2: Contracts, canonical IDs, ADRs — Team A

**Files:**

- Create: `contracts/event-envelope/v0.schema.yaml`
- Create: `contracts/id-registry/v0/entity.schema.yaml`
- Create: `contracts/id-registry/v0/native-assertion.schema.yaml`
- Create: `contracts/id-registry/v0/relation-assertion.schema.yaml`
- Create: `contracts/model-output/v0.schema.yaml`
- Create: `contracts/quality-flags/v0.yaml`
- Create: `contracts/market-relations/v0.yaml`
- Create: `contracts/venue-rule-snapshot/v0.schema.yaml`
- Create: `contracts/deterministic-replay/v0.md`
- Create: `src/prediction_market/contracts.py`
- Create: `artifacts/architecture/adr/0001-engine-decision-deferred.md`
- Create: `artifacts/architecture/adr/0002-native-adapter-control-plane-boundary.md`
- Create: `artifacts/architecture/adr/0003-hot-path-boundary.md`
- Create: `artifacts/architecture/adr/0004-event-sourcing.md`
- Create: `artifacts/architecture/adr/0005-fail-closed.md`
- Create: `tests/contracts/test_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
def test_fixed_point_rejects_binary_float():
    with pytest.raises(ValueError, match="binary float"):
        FixedPointV0.from_value(0.5)

def test_derived_event_requires_parent_and_experiment():
    with pytest.raises(ValidationError):
        EventEnvelopeV0.model_validate(derived_event_without_lineage)

def test_rule_snapshot_requires_raw_hash_and_market_condition():
    snapshot = VenueRuleSnapshotV0.model_validate(valid_snapshot)
    assert snapshot.raw_response_hash.startswith("sha256:")
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/contracts/test_contracts.py -q`  
Expected: missing contract models.

- [ ] **Step 3: Implement the v0 models and machine-readable schemas**

Implement `FixedPointV0(atoms: str, scale: int)`, source/timestamp/reference/lineage models, `EventEnvelopeV0`, `ModelOutputV0`, and `VenueRuleSnapshotV0`. Enforce lowercase SHA-256, UTC `Z` timestamps, controlled quality flags, no extra top-level fields, and conditional lineage/experiment/rule references.

- [ ] **Step 4: Verify contracts and governance**

Run: `uv run pytest tests/contracts/test_contracts.py tests/test_governance.py -q`  
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add contracts src/prediction_market/contracts.py artifacts/architecture/adr tests/contracts
git commit -m "feat: publish Team A contract suite v0"
```

## Task 3: Experiment registry and validation standard — Team H

**Files:**

- Create: `registries/experiments/X-01.yaml` through `X-10.yaml`
- Create: `registries/experiment_registry.csv`
- Create: `registries/artifact_registry.csv`
- Create: `artifacts/validation/validation_standard_v0.md`
- Create: `src/prediction_market/experiments.py`
- Create: `tests/test_experiment_registry.py`

- [ ] **Step 1: Write failing registry tests**

```python
def test_all_seed_experiments_are_registered(program_root):
    registry = load_experiment_registry(program_root)
    assert set(registry) == {f"X-{n:02d}" for n in range(1, 11)}

def test_result_is_rejected_without_preexisting_registration(tmp_path):
    with pytest.raises(UnregisteredExperimentError):
        validate_result_ref(tmp_path, "X-99", "sha256:" + "0" * 64)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_experiment_registry.py -q`  
Expected: registry loader missing.

- [ ] **Step 3: Register X-01 through X-10 without fabricating unresolved parameters**

Cards must preserve Charter hypotheses, metrics, gates, dependencies, and NO-GO restrictions. Unresolved pre-run choices are explicit `registration_locks`; `execution_authorized` is false for X-05, formal X-07, Kalshi X-08, and X-10 recall until their stated locks clear. X-08 uses the prospective recorder-continuity hypothesis.

- [ ] **Step 4: Verify registry, contracts, and source lineage**

Run: `uv run pytest tests/test_experiment_registry.py tests/contracts/test_contracts.py -q`  
Expected: all tests pass.  
Run: `uv run python tools/validate_governance.py`  
Expected: governance valid and ten experiments registered.

- [ ] **Step 5: Commit**

```bash
git add registries artifacts/validation src/prediction_market/experiments.py tests/test_experiment_registry.py
git commit -m "feat: register X-01 through X-10"
```

## Task 4: Immutable raw event store — Team C

**Files:**

- Create: `contracts/raw-capture/v0.schema.yaml`
- Create: `src/prediction_market/raw_store.py`
- Create: `tests/data/test_raw_store.py`
- Create: `data/raw/README.md`
- Create: `data/manifests/README.md`
- Create: `data/normalized/README.md`

- [ ] **Step 1: Write failing immutability tests**

```python
def test_sealed_segment_preserves_exact_payload_and_hash(tmp_path):
    segment = RawSegmentWriter(tmp_path, source="polymarket", stream="market")
    segment.append(b'{"event":"book"}', receive_at=UTC_TIME)
    manifest = segment.seal()
    assert manifest.record_count == 1
    assert verify_segment(manifest.path).valid

def test_sealed_segment_cannot_be_reopened(tmp_path):
    writer = RawSegmentWriter(tmp_path, source="polymarket", stream="market")
    writer.seal()
    with pytest.raises(ImmutableSegmentError):
        writer.append(b"late", receive_at=UTC_TIME)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/data/test_raw_store.py -q`  
Expected: raw-store types missing.

- [ ] **Step 3: Implement staging, Zstandard capture, atomic seal, and manifest verification**

Each record stores base64 payload, exact payload SHA-256, session ID, ordinal, and receive time. `seal()` writes outside the raw prefix, fsyncs, calculates exact-object SHA-256, writes the immutable manifest, then atomically publishes both. Existing final paths must raise and never overwrite.

- [ ] **Step 4: Verify RED/GREEN behavior and full suite**

Run: `uv run pytest tests/data/test_raw_store.py -q`  
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add contracts/raw-capture src/prediction_market/raw_store.py tests/data data
git commit -m "feat: add immutable raw capture store"
```

## Task 5: Public venue recorders and venue-rule snapshots — Team B+C

**Files:**

- Create: `src/prediction_market/adapters/base.py`
- Create: `src/prediction_market/adapters/polymarket.py`
- Create: `src/prediction_market/adapters/kalshi.py`
- Create: `src/prediction_market/recording.py`
- Create: `src/prediction_market/cli/record_markets.py`
- Create: `tests/adapters/test_polymarket.py`
- Create: `tests/adapters/test_kalshi.py`
- Create: `tests/test_recording.py`
- Create: `artifacts/venue-connectivity/capability_matrix_v0.csv`
- Create: `artifacts/venue-connectivity/kalshi_credentials.md`

- [ ] **Step 1: Write failing protocol/auth/reconnect tests**

```python
async def test_polymarket_frames_are_written_before_parsing(fake_ws, raw_root):
    await record_polymarket(fake_ws, ["asset-1"], raw_root, max_frames=1)
    assert read_first_payload(raw_root) == fake_ws.frames[0]

def test_kalshi_signature_is_rsa_pss_sha256(private_key):
    headers = kalshi_auth_headers("GET", "/trade-api/ws/v2", 1234, "key", private_key)
    verify_pss_signature(headers, private_key.public_key())
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/adapters tests/test_recording.py -q`  
Expected: adapters missing.

- [ ] **Step 3: Implement native adapters and fail-closed rule snapshot capture**

Polymarket public market WebSocket needs no credential. Kalshi loads the key ID and RSA private-key path from environment references and never logs key material. Adapters capture raw bytes before parsing, use bounded reconnect backoff, expose gap/reconnect counters, and store every rule response before normalization. Missing rule fields create an invalid quality-marked snapshot that formal replay must reject.

- [ ] **Step 4: Run tests and a bounded public smoke capture**

Run: `uv run pytest tests/adapters tests/test_recording.py -q`  
Expected: all tests pass.  
Run: `uv run record-markets polymarket --discover-sports --max-frames 20 --raw-root var/raw`  
Expected: at least one sealed manifest when active public markets are available; otherwise a nonzero exit with a specific discovery/connection reason and no fabricated data.

- [ ] **Step 5: Commit**

```bash
git add src/prediction_market/adapters src/prediction_market/recording.py src/prediction_market/cli tests/adapters tests/test_recording.py artifacts/venue-connectivity
git commit -m "feat: add native public market recorders"
```

## Task 6: PMXT Phase 0/1 and deterministic L2 reconstruction — Team C

**Files:**

- Create: `src/prediction_market/pmxt/archive.py`
- Create: `src/prediction_market/pmxt/reconstructor.py`
- Create: `src/prediction_market/pmxt/quality.py`
- Create: `src/prediction_market/cli/audit_pmxt.py`
- Create: `tests/pmxt/test_reconstructor.py`
- Create: `tests/pmxt/test_archive.py`
- Create: `tests/fixtures/pmxt/l2_events.jsonl`
- Create: `artifacts/data-audit/phase0_coverage_cost_v0.md`
- Create: `artifacts/data-audit/phase1_schema_timestamp_v0.md`
- Create: `artifacts/data-audit/l2_reconstructor_v0.md`

- [ ] **Step 1: Write failing reconstruction tests**

```python
def test_reconstruction_is_level_2_deterministic(pmxt_fixture):
    first = reconstruct(pmxt_fixture)
    second = reconstruct(pmxt_fixture)
    assert first.semantic_events == second.semantic_events
    assert first.stream_sha256 == second.stream_sha256

def test_reconstructor_flags_crossed_and_nonpositive_books(anomalous_fixture):
    result = reconstruct(anomalous_fixture)
    assert {"CROSSED_BOOK", "NONPOSITIVE_SIZE"} <= set(result.quality_flags)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/pmxt -q`  
Expected: PMXT package missing.

- [ ] **Step 3: Implement archive inventory, schema audit, fixed-point book application, dedupe, gaps, and canonical hashing**

Preserve original Parquet files and their hashes. Merge globally by the Charter key, use explicit `ORDER BY`, count duplicate/out-of-order/gap/anomaly classes, and never claim queue position or fill reconstruction.

- [ ] **Step 4: Run synthetic tests and a bounded public Phase 0/1 sample**

Run: `uv run pytest tests/pmxt -q`  
Expected: all tests pass.  
Run: `uv run audit-pmxt inventory --output artifacts/data-audit/phase0_inventory.json`  
Expected: measured coverage/file-size evidence or a documented network/source failure.  
Run: `uv run audit-pmxt sample --max-files 1 --raw-root var/raw`  
Expected: original file hash, measured schema, and reconstruction report when a sample is reachable.

- [ ] **Step 5: Commit**

```bash
git add src/prediction_market/pmxt src/prediction_market/cli/audit_pmxt.py tests/pmxt tests/fixtures/pmxt artifacts/data-audit
git commit -m "feat: implement PMXT phase zero and one audit"
```

## Task 7: Validation metrics and D1/D2/D3 baseline POCs — Team D1/D2/D3+H

**Files:**

- Create: `src/prediction_market/models/validation.py`
- Create: `src/prediction_market/models/nba.py`
- Create: `src/prediction_market/models/nfl.py`
- Create: `src/prediction_market/models/soccer.py`
- Create: `tests/models/test_validation.py`
- Create: `tests/models/test_baselines.py`
- Create: `artifacts/game-state/nba/baseline_v0.md`
- Create: `artifacts/game-state/nfl/evidence_pipeline_poc_v0.md`
- Create: `artifacts/game-state/soccer/evidence_pipeline_poc_v0.md`

- [ ] **Step 1: Write failing grouped-split/calibration tests**

```python
def test_walk_forward_never_splits_a_game_or_uses_future_games():
    folds = list(game_grouped_walk_forward(frame, min_train_games=2))
    assert all(set(train.game_id).isdisjoint(test.game_id) for train, test in folds)
    assert all(train.played_at.max() < test.played_at.min() for train, test in folds)

def test_metric_report_contains_required_calibration_and_cluster_ci():
    report = evaluate_probabilities(y, prior, logistic, groups=game_ids, bootstrap=200)
    assert {"brier", "log_loss", "slope", "intercept", "bootstrap_ci"} <= report.keys()
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/models -q`  
Expected: model modules missing.

- [ ] **Step 3: Implement market-prior, logistic, and histogram-GBDT baselines with PIT feature contracts**

All three sports share grouped chronological validation and calibration reporting. NBA compares prior/logistic/GBDT. NFL and soccer produce evidence/data-pipeline POCs and calibration results only when their own H-issued experiment IDs are present; otherwise the CLI refuses to write result artifacts.

- [ ] **Step 4: Verify models**

Run: `uv run pytest tests/models -q`  
Expected: all tests pass with deterministic seeds.

- [ ] **Step 5: Commit**

```bash
git add src/prediction_market/models tests/models artifacts/game-state
git commit -m "feat: add calibrated game-state baseline POCs"
```

## Task 8: Barrier-hit label specification and generator — Team E+H

**Files:**

- Create: `artifacts/reaction-labels/x05_barrier_label_spec_v0.md`
- Create: `src/prediction_market/labels.py`
- Create: `tests/test_labels.py`

- [ ] **Step 1: Write failing executable-quote tests**

```python
def test_long_label_enters_at_ask_and_exits_at_bid():
    label = label_long(quotes, anchor=ANCHOR, upper=UPPER, lower=LOWER, horizon=HORIZON)
    assert label.entry_price == quotes[0].ask
    assert label.exit_price in {q.bid for q in quotes}

def test_midpoint_and_suspended_quotes_are_never_label_prices():
    label = label_long(quotes_with_suspension, **locked_parameters)
    assert label.entry_price != midpoint(quotes_with_suspension[0])
    assert label.resume_quote_id == FIRST_EXECUTABLE_AFTER_RESUME
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_labels.py -q`  
Expected: label module missing.

- [ ] **Step 3: Implement parameterized label generation with no unregistered defaults**

The generator requires explicit U/L/h, quote-staleness, overlap, purge, and embargo parameters. It cannot run from X-05 until the card is amended and H authorizes execution. Suspended quotes are ineligible; the first non-stale two-sided quote after resume is the first executable quote.

- [ ] **Step 4: Verify label rules**

Run: `uv run pytest tests/test_labels.py -q`  
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add artifacts/reaction-labels src/prediction_market/labels.py tests/test_labels.py
git commit -m "feat: specify executable barrier labels"
```

## Task 9: Taker simulator, TCA, and maker bounds — Team F

**Files:**

- Create: `contracts/tca/v0.schema.yaml`
- Create: `artifacts/execution-tca/taker_simulator_spec_v0.md`
- Create: `artifacts/execution-tca/maker_bounds_v0.md`
- Create: `src/prediction_market/execution.py`
- Create: `tests/test_execution.py`

- [ ] **Step 1: Write failing taker-realism tests**

```python
def test_buy_consumes_ask_depth_after_rule_and_system_delay():
    fill = simulate_taker_buy(book_stream, order, rule_snapshot, own_delay_ms=500)
    assert fill.executed_at >= order.created_at + rule_snapshot.delay + timedelta(milliseconds=500)
    assert fill.vwap == expected_ask_vwap

def test_formal_result_rejects_missing_rule_snapshot():
    with pytest.raises(PreliminaryOnlyError):
        simulate_taker_buy(book_stream, order, None, own_delay_ms=500, formal=True)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_execution.py -q`  
Expected: execution module missing.

- [ ] **Step 3: Implement ask/bid, depth-VWAP, fee, delay, markout, and TCA records**

Only taker execution is simulated. Maker output is limited to optimistic/base/pessimistic bounds and contains no trained queue point estimate. Every formal simulation references the exact venue-rule event and emits post-fill markout fields.

- [ ] **Step 4: Verify execution**

Run: `uv run pytest tests/test_execution.py -q`  
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add contracts/tca artifacts/execution-tca src/prediction_market/execution.py tests/test_execution.py
git commit -m "feat: add taker simulator and TCA v0"
```

## Task 10: Matched-cluster audit and eight module reports — Team G+A

**Files:**

- Create: `src/prediction_market/clusters.py`
- Create: `tests/test_clusters.py`
- Create: `artifacts/strategy-reports/x10_matched_cluster_audit_v0.md`
- Create: eight reports under `artifacts/strategy-reports/modules/`

- [ ] **Step 1: Write failing precision/review-queue tests**

```python
def test_precision_gate_requires_at_least_45_of_50():
    assert cluster_gate(correct=45, reviewed=50).may_advance is True
    assert cluster_gate(correct=44, reviewed=50).may_advance is False

def test_recall_is_refused_without_candidate_universe():
    with pytest.raises(MissingRecallDenominatorError):
        compute_recall(reviewed_matches, candidate_universe=None)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_clusters.py -q`  
Expected: cluster module missing.

- [ ] **Step 3: Implement audit import, relation taxonomy validation, confidence bins, and review queue**

Reports cover cross-venue, arbitrage, copy, whale, news, LLM, weather, and forecasting-agent modules. Each includes evidence, real execution difficulty, honest-backtest design, and a no-demo/no-live restriction. LLM remains slow-path only.

- [ ] **Step 4: Verify cluster workflow**

Run: `uv run pytest tests/test_clusters.py -q`  
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/prediction_market/clusters.py tests/test_clusters.py artifacts/strategy-reports
git commit -m "feat: add matched-cluster and module audit workflow"
```

## Task 11: Compliance, licensing, and secret handling — Team I

**Files:**

- Create: `registries/compliance_matrix.csv`
- Create: `registries/data_license_register.csv`
- Create: `artifacts/compliance/secret_management_v0.md`
- Create: `artifacts/compliance/production_readiness_v0.md`
- Create: `tests/test_compliance_registry.py`

- [ ] **Step 1: Write failing hard-gate tests**

```python
def test_real_money_is_blocked_when_any_required_compliance_row_is_not_green(program_root):
    matrix = load_compliance_matrix(program_root)
    assert may_execute_real_money(matrix) is False

def test_operational_evidence_ids_are_all_present(program_root):
    register = load_license_register(program_root)
    assert {row.catalog_item_id for row in register} == {f"O-{n:03d}" for n in range(1, 9)}
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_compliance_registry.py -q`  
Expected: compliance registry missing.

- [ ] **Step 3: Populate evidence-dated rows from authoritative terms without granting green status**

Rows record jurisdiction/account/license unknowns explicitly. No row becomes green without operating-jurisdiction and account-context evidence. Secret handling uses key references and least privilege; private keys and wallet material are never committed.

- [ ] **Step 4: Verify compliance hard gate**

Run: `uv run pytest tests/test_compliance_registry.py -q`  
Expected: all tests pass and real-money remains blocked.

- [ ] **Step 5: Commit**

```bash
git add registries/compliance_matrix.csv registries/data_license_register.csv artifacts/compliance tests/test_compliance_registry.py
git commit -m "docs: establish Team I compliance gates"
```

## Task 12: X-09 deterministic vertical slice — Team B+A+H

**Files:**

- Create: `src/prediction_market/replay.py`
- Create: `src/prediction_market/vertical_slice.py`
- Create: `tests/replay/test_determinism.py`
- Create: `tests/replay/test_vertical_slice.py`
- Create: `artifacts/architecture/x09_vertical_slice_v0.md`

- [ ] **Step 1: Write failing Level 1/2 tests**

```python
def test_vertical_slice_has_identical_semantics_and_hash_twice(frozen_input):
    first = run_vertical_slice(frozen_input)
    second = run_vertical_slice(frozen_input)
    assert first.semantic_summary == second.semantic_summary
    assert first.stream_sha256 == second.stream_sha256

def test_every_simulated_order_has_trigger_lineage(frozen_input):
    result = run_vertical_slice(frozen_input)
    assert all(order.parent_event_ids for order in result.orders)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/replay -q`  
Expected: replay modules missing.

- [ ] **Step 3: Implement explicit ordering, canonical JSON, dummy signal, risk check, taker fill, P&L, and event lineage**

The slice uses only simulated orders and exact fixed-point arithmetic. It consumes registered X-09 configuration, one rule snapshot, and a frozen fixture. Any semantic or hash divergence is a hard failure and leaves all strategy backtests frozen.

- [ ] **Step 4: Verify Level 1+2 twice**

Run: `uv run pytest tests/replay -q`  
Expected: all tests pass and both runs produce the same 64-character hash.

- [ ] **Step 5: Commit**

```bash
git add src/prediction_market/replay.py src/prediction_market/vertical_slice.py tests/replay artifacts/architecture/x09_vertical_slice_v0.md
git commit -m "feat: pass X-09 deterministic vertical slice"
```

## Task 13: Program integration and final audit — All teams

**Files:**

- Create: `artifacts/architecture/week1_backlog_v0.csv`
- Create: `artifacts/architecture/risk_blocker_register_v0.csv`
- Create: `artifacts/architecture/program_status_v0.md`
- Modify: `registries/artifact_registry.csv`
- Modify: `README.md`

- [ ] **Step 1: Add integration assertions**

```python
def test_all_first_round_artifacts_have_owner_version_and_due_gate(program_root):
    rows = load_artifact_registry(program_root)
    assert rows
    assert all(row.owner and row.version and row.due_gate for row in rows)

def test_no_go_audit_remains_closed(program_root):
    assert audit_no_go(program_root).violations == ()
```

- [ ] **Step 2: Run complete verification**

Run: `uv run pytest -q`  
Expected: zero failures.  
Run: `uv run python tools/validate_governance.py`  
Expected: source/catalog/experiment/artifact/compliance checks pass.  
Run: `git status --short`  
Expected: only intentional final integration files before commit.

- [ ] **Step 3: Publish actual status without overstating time- or credential-gated work**

The status report distinguishes complete artifacts, running public capture, synthetic/fixture verification, pending seven-day X-08 evidence, pending Kalshi credentials, unresolved X-05 locks, X-07 `PRELIMINARY`, X-10 recall lock, and Team I non-green rows.

- [ ] **Step 4: Commit integration state**

```bash
git add README.md registries/artifact_registry.csv artifacts/architecture
git commit -m "docs: publish first-round program execution status"
```

- [ ] **Step 5: Request final spec and code-quality review**

Review every direct user requirement and Charter NO-GO item against the implementation, then run the complete verification command again after all review fixes.

