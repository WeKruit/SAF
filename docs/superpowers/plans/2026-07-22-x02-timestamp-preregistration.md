# X-02 Timestamp Preregistration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one append-only Team H amendment that fixes the X-02 sample, timestamp statistics, downgrade decision, and audit-split approval, then add a second append-only amendment that binds the completed four-day input bundle without preregistering runner code or evaluated data.

**Architecture:** Store the exact preregistration as an X-02-only controlled amendment and couple its three resolved locks to canonical hashes of the three structured sections. Bind the later manifest bundle through a separate exact controlled change that resolves only `timestamp_input_manifest` after verifying the bundle, its four daily manifests, 96 hourly objects, and their static-manifest references. Preserve the trusted base record and append both amendments to the separate ledger. Update the normative validation standard and its X-05 content-addressed dependency as one hash chain.

**Tech Stack:** Python 3.12, PyYAML, canonical JSON SHA-256, CSV registries, pytest.

**Execution constraint:** Do not stage or commit; the parent task owns final staging and commit.

---

### Task 1: Specify the append-only contract with failing tests

**Files:**
- Modify: `tests/test_experiment_registry.py`

- [ ] **Step 1: Add an exact expected preregistration fixture**

```python
EXPECTED_X02_PREREGISTRATION = {
    "sampling_and_seed": {
        "selection_seed": 20260722,
        "game_day": "2026-05-28",
        "random_days": ["2026-04-22", "2026-06-05", "2026-06-25"],
        "selection_inventory_sha256": "sha256:74d7e9f21003f595d2d505bc63c89c7eadcd5339d541032bc80704d4b14b3043",
        "x01_game_day_manifest_sha256": "sha256:e7dfc9e7992f1eb085edc0c67f37100db10c6541533220833dcace2a1e244df3",
        "pending_archive_object_count": 72,
    },
    "diff_and_stability_definitions": {
        "diff_ms": "epoch_ms(timestamp_received)-epoch_ms(timestamp)",
        "signed_quantiles": {
            "estimator": "quantile_cont",
            "interpolation": "linear",
            "input_distribution": "integer_millisecond_frequency",
            "probabilities": ["0.50", "0.95", "0.99"],
        },
        "absolute_p99": {
            "transform": "abs(diff_ms)",
            "estimator": "quantile_cont",
            "interpolation": "linear",
            "input_distribution": "integer_millisecond_frequency",
            "probability": "0.99",
        },
        "hourly_drift_ms": "median_utc_hour_23_diff_ms-median_utc_hour_00_diff_ms",
        "disorder": {
            "partition_by": ["market", "asset_id"],
            "canonical_sort": ["timestamp_received", "timestamp"],
            "numerator": "adjacent_source_timestamp_strict_descents",
            "ordered_comparisons": "n_rows-n_unique_market_asset_streams",
        },
        "downgrade_gate": {
            "negative_rate_gte": "0.001",
            "absolute_p99_ms_gt": 5000,
            "decision": "downgrade_millisecond_research_to_seconds",
        },
    },
    "h_split_approval": {
        "split": "n.a.",
        "basis": "charter_section_9_measurement_audit_exemption",
        "approved_by": "H",
    },
}
```

- [ ] **Step 2: Add seed-chain and formal-block tests**

Assert that sequence 0 remains the original X-02 base hash, sequence 1 contains the exact fixture, resolves the three corresponding locks with each section’s canonical hash, leaves `timestamp_input_manifest` unresolved, and still rejects a formal result.

- [ ] **Step 3: Add fail-closed mutation tests**

Parameterize mutations of the seed, dates, both source hashes, object count, diff sign, quantile estimator/interpolation/distribution, absolute-P99 transform, drift sign, disorder sort/numerator/denominator, downgrade comparison/threshold, split basis, evidence hash, missing lock, extra lock, missing structured change, duplicate structured change, result co-mutation, and non-H approval.

- [ ] **Step 4: Run the new tests and verify RED**

Run:

```bash
uv run pytest -q \
  tests/test_experiment_registry.py::test_x02_seed_preregistration_is_exact_and_keeps_input_manifest_locked \
  tests/test_experiment_registry.py::test_x02_formal_result_remains_blocked_until_input_manifest_bundle \
  tests/test_experiment_registry.py -k 'x02_preregistration'
```

Expected: failures because the card has no seed amendment and the validator does not recognize `timestamp_audit_preregistration`.

### Task 2: Implement exact X-02 amendment validation

**Files:**
- Modify: `src/prediction_market/experiments.py`

- [ ] **Step 1: Add the exact immutable X-02 preregistration constant**

Define `_X02_TIMESTAMP_AUDIT_PREREGISTRATION` with exactly the structure in Task 1. Do not accept alternate strings, numeric encodings, additional keys, or omitted keys.

- [ ] **Step 2: Add the controlled change key**

Add `timestamp_audit_preregistration` to `_validate_changes`. Reject it for experiments other than X-02, reject `status` or `results_ref` in the same amendment, and exact-compare the payload to `_X02_TIMESTAMP_AUDIT_PREREGISTRATION`.

- [ ] **Step 3: Couple lock resolution to structured evidence**

During `_apply_amendments`, require the structured change to accompany exactly these lock IDs:

```python
{
    "sampling_and_seed",
    "diff_and_stability_definitions",
    "h_split_approval",
}
```

Require each `evidence_ref` to equal:

```python
"sha256:" + hashlib.sha256(
    _canonical_bytes(preregistration[lock_id])
).hexdigest()
```

Reject independent resolution of any of those locks, reject resolution of `timestamp_input_manifest` in the seed amendment, enforce append-once behavior, and expose a deep-copied `timestamp_audit_preregistration` on the effective X-02 card.

- [ ] **Step 4: Run the new tests and verify GREEN**

Run:

```bash
uv run pytest -q tests/test_experiment_registry.py -k 'x02_preregistration'
```

Expected: all selected tests pass.

### Task 3: Append the seed record and synchronize governance hashes

**Files:**
- Modify: `registries/experiments/X-02.yaml`
- Modify: `registries/experiment_amendment_ledger.csv`
- Modify: `registries/experiment_registry.csv`
- Modify: `artifacts/validation/validation_standard_v0.md`
- Modify: `registries/experiments/X-05.yaml`
- Modify: `src/prediction_market/experiments.py`

- [ ] **Step 1: Compute the three section hashes**

Use canonical JSON bytes with sorted keys and compact separators. Insert those hashes into the three `resolve_locks` rows of X-02 sequence 1.

- [ ] **Step 2: Append X-02 sequence 1**

Use:

```yaml
sequence: 1
amended_at: '2026-07-23T05:15:58Z'
prior_sha256: sha256:1b6393ef8cca4bf482cc3a167844358c07fda9a97c45cbedbe4ceda3e2033ed1
approved_by: H
reason: Preregister the X-02 sample, timestamp statistics, downgrade gate, and Charter-approved audit split before formal data inspection.
```

Set `changes.resolve_locks` and `changes.timestamp_audit_preregistration`, compute the amendment SHA with `compute_amendment_sha256`, append the exact row to the ledger, and leave `registration_record_sha256` unchanged.

- [ ] **Step 3: Update normative text**

Document the X-02-only controlled amendment, exact sample, exact signed and absolute quantile definitions, drift/disorder formulas, downgrade gate, and unresolved input-manifest requirement in `validation_standard_v0.md`.

- [ ] **Step 4: Resynchronize content-addressed dependencies**

Compute the new validation-standard SHA, update X-05’s artifact dependency, recompute X-05’s base registration hash, update its trusted map entry and ledger sequence 0, then recompute X-02 and X-05 card-file hashes in `experiment_registry.csv`. X-02’s trusted base hash and ledger sequence 0 must not change.

- [ ] **Step 5: Adjust existing X-02 fixture timestamps**

Move test-only X-02 amendments later than the new seed amendment without changing the intended dependency-order assertions.

### Task 4: Verify the full governance chain

**Files:**
- Test: `tests/test_experiment_registry.py`
- Test: `tests/test_research_registries.py`

- [ ] **Step 1: Run focused tests**

```bash
uv run pytest -q \
  tests/test_experiment_registry.py \
  tests/test_research_registries.py \
  tests/test_compliance_registry.py
```

Expected: all pass.

- [ ] **Step 2: Run the full suite**

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run integrity checks**

```bash
git diff --check
(cd charter && shasum -a 256 -c SOURCE_MANIFEST.sha256)
rg -n 'O-00''9|O-01''[0-2]' . --glob '!**/.git/**'
```

Expected: no diff errors, all Charter source hashes `OK`, and no removed unstable O IDs.

### Task 5: Bind the completed four-day input bundle

**Files:**
- Modify: `tests/test_experiment_registry.py`
- Modify: `src/prediction_market/experiments.py`
- Modify: `registries/experiments/X-02.yaml`
- Modify: `registries/experiment_amendment_ledger.csv`
- Modify: `registries/experiment_registry.csv`
- Modify: `registries/artifact_registry.csv`
- Modify: `artifacts/validation/validation_standard_v0.md`
- Modify: `registries/experiments/X-05.yaml`

- [ ] **Step 1: Add RED tests for sequence 2**

Assert the exact path, file hash, bundle self-hash, single resolved lock, absent `preregistered_inputs`, and exact ledger chain. Add direct verifier tests for wrong path/file/self hash, wrong day set/order/count, wrong total object count, daily file/self-hash tampering, incomplete daily coverage, invalid static-manifest references, missing files, symlinks, and path traversal.

- [ ] **Step 2: Implement the exact bundle verifier**

Add an X-02-only `timestamp_input_manifest_binding` change. Require exact structured binding, verify the registered bundle path under the program root, verify the file and self hashes, require the exact seed/inventory/selection fields, require the four exact daily manifest records and 96 objects, verify each daily artifact file hash and self-hash, and validate each full-day manifest including all static-manifest references.

- [ ] **Step 3: Append sequence 2 and artifact rows**

Append an H-approved sequence 2 at the actual write time `2026-07-23T05:16:12Z`, with `prior_sha256` equal to sequence 1 and lock evidence equal to the bundle self-hash. Register the three new day manifests and the bundle as `C+H`, `v1`, `2026-08-05_W2_review`, `registered`. Do not duplicate the existing X-01 day file.

- [ ] **Step 4: Keep formal execution blocked**

Do not append code/data preregistration, results, or status. Assert that the effective manifest lock is resolved while `preregistered_inputs` remains empty and a formal result is rejected for missing preregistered inputs.

- [ ] **Step 5: Resynchronize and verify**

Update the validation standard, then resynchronize X-05’s content-addressed dependency/base/trust/ledger chain and both card-file hashes. Run the focused registry suite, full suite, and integrity checks.
