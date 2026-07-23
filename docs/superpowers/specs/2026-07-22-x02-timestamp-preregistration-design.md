# X-02 Timestamp Audit Preregistration Design

## Goal

Append two narrowly scoped Team H amendments to X-02 before formal data inspection. Sequence 1 fixes the sample, statistics, downgrade gate, and `n.a.` split approval without rewriting the immutable base registration. Sequence 2 binds the completed four-day input bundle and resolves only `timestamp_input_manifest`; it does not preregister runner code or evaluated data.

## Amendment contract

The X-02-only `timestamp_audit_preregistration` change has three exact sections:

1. `sampling_and_seed`
   - selection seed: `20260722`
   - game day: `2026-05-28`
   - random days: `2026-04-22`, `2026-06-05`, `2026-06-25`
   - selection inventory: `sha256:74d7e9f21003f595d2d505bc63c89c7eadcd5339d541032bc80704d4b14b3043`
   - known X-01 full-day manifest: `sha256:e7dfc9e7992f1eb085edc0c67f37100db10c6541533220833dcace2a1e244df3`
   - pending archive objects for the other three daily manifests: `72`
2. `diff_and_stability_definitions`
   - `diff_ms = epoch_ms(timestamp_received) - epoch_ms(timestamp)`
   - signed P50/P95/P99 use exact `quantile_cont` linear interpolation over the integer-millisecond frequency distribution
   - absolute P99 is computed separately by applying the same estimator to `abs(diff_ms)`
   - hourly drift is `median(diff_ms where UTC hour = 23) - median(diff_ms where UTC hour = 00)`
   - disorder partitions by `(market, asset_id)`, sorts canonically ascending by `(timestamp_received, timestamp)`, and counts adjacent pairs whose source `timestamp` strictly decreases
   - ordered-comparison denominator is `n_rows - n_unique_market_asset_streams`
   - downgrade is triggered when `negative_rate >= 0.001` or `absolute_p99_ms > 5000`
3. `h_split_approval`
   - split is exactly `n.a.`
   - basis is the Charter §9 measurement-audit exemption
   - approver is `H`

The same amendment resolves exactly `sampling_and_seed`, `diff_and_stability_definitions`, and `h_split_approval`. Each lock’s `evidence_ref` must equal the canonical SHA-256 of its corresponding section. The change is append-once, H-approved, and cannot share an amendment with a result or status transition.

## Fail-closed behavior

The three locks cannot be resolved through an opaque `resolve_locks` amendment. They resolve only when the exact structured preregistration is present and its evidence hashes match. Any changed seed, date, inventory hash, manifest hash, estimator, interpolation rule, distribution, drift sign, disorder numerator/denominator, threshold, split, or approver is rejected.

`timestamp_input_manifest` remains unresolved after the seed amendment. Once the 72 pending objects and three missing daily manifests exist, sequence 2 may bind only:

- path `artifacts/data-audit/x02_timestamp_input_bundle_v1.json`
- file SHA-256 `sha256:46c3f23007929ad31b131f3618009810501b6ef06d0ac2645eb3dcfac217bd8d`
- bundle self-hash `sha256:9477f8d9a224b47b6dda47dd691761d4ca8b5d88be6ad07416217a2bb44c89a4`

The verifier must load that file within the program root, verify its file and self hashes, require the exact seed/inventory/day selection, require four ordered daily manifests and 96 total hourly objects, verify every daily artifact file hash and self-hash, validate every daily manifest as a complete ordered UTC day, and open and validate all 96 tracked static-manifest sidecars. Each sidecar path is derived from its hourly object path and static-manifest digest; the sidecar must be a canonical, non-symlinked `StaticDatasetManifestV0` whose self-hash, native object path, object hash, source URL, partition, dataset, and approved license match the daily-manifest object. `inventory_size_bytes` is an archive-listing estimate and must not be reported as an exact object byte length; exact object bytes are stated only by the validated static sidecar’s `byte_length`. The lock evidence is the bundle self-hash.

Sequence 2 contains only `resolve_locks` for `timestamp_input_manifest` and the structured bundle binding. It cannot carry `preregistered_inputs`, a result, or a status transition. The formal result remains blocked until an independent later amendment freezes reviewed runner-code and evaluated-data hashes.

## Persistence and verification

The original X-02 registration record and ledger sequence 0 remain unchanged. Sequence 1 points to the trusted base hash. Sequence 2 points to sequence 1. Both are duplicated exactly in the amendment ledger. Tests cover the exact seed record, coupled lock resolution, exact bundle binding, file and self-hash verification, four-day/96-object verification, daily manifest verification, static-reference verification, tampering, omission, duplicate structured changes, unchanged base hash, absent code/data preregistration, and formal-result rejection.

The validation standard’s controlled-amendment list and X-02 gate text are updated in the same change. Because X-05 content-addresses that standard, its artifact dependency, immutable base hash, trust anchor, ledger seed, and card-file registry hash are resynchronized as one chain.
