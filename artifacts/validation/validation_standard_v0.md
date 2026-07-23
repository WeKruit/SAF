# Validation Standard v0

- **Owner:** Team H
- **Version:** v0
- **Program state:** `CONDITIONAL_GO`
- **Source of truth:** `charter/research_program_charter_v0.2.md`, `catalog_registry.csv`, `catalog_team_assignments.csv`
- **Due gate:** `first_round_artifact_gate`

This standard operationalizes the Charter. It does not alter approved scope, promotion gates, or NO-GO decisions.

## 1. Registration is a hard prerequisite

Only X-01 through X-12 exist in registry v0. A result is invalid when its experiment, result scope, exact dataset/model IDs, code hash, or input-data hash was not registered by a valid Team H amendment before evaluation began. The result-artifact hash is necessarily created after evaluation: it must be a canonical SHA-256 reference and is appended to `results_ref` by a later Team H amendment against the exact registration head used for evaluation. An unknown experiment remains invalid even if it supplies a syntactically valid hash. “Quick backtests” outside the registry have no evidentiary value.

The base card is content-addressed and immutable. Its registration SHA is pinned both in validator code and at sequence 0 of `registries/experiment_amendment_ledger.csv`; recomputing a card self-hash or its CSV file hash cannot rewrite that trusted seed. Corrections are append-only amendments with a contiguous sequence, the preceding record hash, an amendment content hash, an exact `H` or `Team H` approver, canonical UTC time, reason, and controlled changes. The card chain and separate ledger must match exactly in both directions. Git protected history is the external monotonic anchor; inside any checkout, a ledger/card mismatch fails closed. `failed` and `abandoned` records are retained permanently; history is never overwritten.

Amendments may change status, append a result reference, resolve named locks with content-addressed evidence, authorize a non-permanent scope, or preregister code/data hashes for a named scope. X-02 alone may append `timestamp_audit_preregistration`, and only when that amendment contains the exact H-approved sample, definitions, split approval, and canonical section hashes for its three preregistration locks; `timestamp_input_manifest` is not part of that resolution. A separate X-02 `timestamp_input_manifest_binding` amendment may resolve only that remaining lock: it must bind the exact registered four-day bundle path, file SHA-256, and self-hash, and the validator must verify all four daily artifact files, daily self-hashes, 96 ordered hourly objects, and 96 static-manifest references. That bundle amendment may not carry code/data preregistration, a result, or a status change. X-08 alone may append `archive_audit_clarification` with the exact stopped-archive/reference-audit boundary and append `observed_elapsed_evidence` bound to a registered canonical capture manifest. These experiment-specific changes are append-only and may not replace their Charter-aligned base registrations. Amendments may not overwrite the hypothesis, method, scientific boundary, dependencies, or program NO-GOs. Every accepted result carries `registration_head_sha256`; that value must equal the effective head at evaluation. Evaluation must start strictly after the amendment that preregistered its matching code and data hashes.

Because the source program registration is dated but not time-stamped, v0 conservatively accepts evaluation only from `2026-07-23T00:00:00Z`. This is an evidence-acceptance boundary, not a claim that registration occurred at an invented time.

## 2. Authorization is scope-specific and fail-closed

`execution_authorized` is the experiment-level runtime kill switch; it does not itself accept a complete or formal result. A true sub-scope never authorizes another sub-scope, and every formal or promotion scope marked permanent NO-GO remains closed. In particular, X-11 is limited to its preregistered pipeline result and X-12 is limited to its POC result; neither card can produce a formal or promoted result in v0. Every scope-specific registration lock must be resolved with evidence. A dependency is ready only when it is `done`, has a validated appended result, has matching preregistered code/data, has its result-scope locks resolved, and recursively has ready dependencies; status alone is never evidence.

- X-05 authorizes spec drafting only; label generation and a formal result remain blocked until U/L/h, purge, embargo, overlap, suspension/resume, same-time-touch, and H-signature locks clear.
- X-02 fixes selection seed `20260722`, game day `2026-05-28`, random days `2026-04-22`/`2026-06-05`/`2026-06-25`, the content-addressed selection inventory, the known X-01 day manifest, and 72 pending archive objects before formal inspection. The sampling, statistical-definition, and H audit-split locks are resolved by that exact structured amendment. The later verified bundle `sha256:9477f8d9a224b47b6dda47dd691761d4ca8b5d88be6ad07416217a2bb44c89a4` binds the exact four daily manifests and resolves only `timestamp_input_manifest`. Code and evaluated-data hashes remain unregistered until independent runner review passes, so no X-02 formal evaluation or result is yet admissible.
- Charter §9 makes X-02 depend on the X-01 Phase 1 reconstruction pipeline, not the X-01 Phase 3 independent-comparison gate. Registry v0 has only experiment-terminal dependencies, so its generic acceptance check conservatively blocks X-02 until X-01 is `done`. This representational over-blocking does not prevent the preregistered measurement run once the Phase 1 input artifacts and reviewed runner are frozen; any such output must be labeled `MEASUREMENT_COMPLETED_REGISTRY_ACCEPTANCE_DEPENDENCY_BLOCKED` and retained as non-formal evidence until an H-approved artifact-scoped dependency rule exists or X-01 becomes ready. The base `dependencies: [X-01]` is not rewritten.
- `artifacts/data-audit/polymarket_v1_bounded_sports_extract_v0.json` is `PRELIMINARY_RESEARCH_ONLY`. It uses the approved `DS-POLYMARKET-V1` / `R-039` source for a bounded POC; its Gamma validation remains under pending `DS-POLYMARKET-PUBLIC` / `O-001`. It cannot satisfy an experiment formal-result or promotion gate.
- X-07 authorizes only a result labeled `PRELIMINARY`, after the event/depth input, order-size/VWAP policy, and markout horizons are fixed. The venue-rule replay lock is required for formal conclusions and go/no-go use.
- X-08 authorizes the stopped-archive audit only after its source manifest and Team H n.a.-split approval are fixed, and authorizes public Polymarket capture after recorder configuration. Kalshi capture remains credential-blocked. A dual-venue completion requires a registered canonical capture manifest whose SHA-256 is bound by the amendment; that manifest must bind both exact live dataset streams to immutable raw-store sidecars and objects, and every sidecar, object, record, capture session, and source/stream identity is verified. The fixed `recorder-heartbeat/v0` gap policy permits at most 60 seconds between records independently on each live stream. Segments must be sealed after their last record and before a non-future evidence amendment; elapsed coverage is derived from the overlapping verified stream windows, never from caller-supplied start/end claims. Fixtures and short smoke captures cannot satisfy seven elapsed days. The Charter's `99% <= uptime < 100%` band is unresolved and is never silently passed.
- X-10 authorizes the precision audit only after its matched sample, Router/taxonomy, gold-standard protocol, and Team H n.a.-split approval locks clear. Recall requires a registered candidate-universe denominator. Live multi-venue arbitrage remains a permanent NO-GO regardless of precision.

## 3. Point-in-time and leakage discipline

Every input declares its source, version or explicit unresolved version lock, and point-in-time basis. Raw observations are append-only and hashed; normalized data must be reconstructible from raw data. Present-day venue rules must never backfill historical simulation: replay joins the versioned rule stream as of simulated time.

The default predictive split is game-grouped, chronological, and walk-forward. Any `n.a.` audit split requires an H approval lock. No game may cross folds. Purge and embargo parameters are registered before inspection. X-04 treats sports event time as an interval, not a falsely precise production timestamp.

Formal labels and execution evidence use executable ask for entry and bid for exit. Midpoint is forbidden. Suspended quotes are ineligible; the first executable quote after resume must be defined before X-05 label generation.

## 4. Experiment-specific gates

- **X-01 / X-09:** Replay Levels 1 and 2 are mandatory. Equal P&L with a different event order or canonical hash fails. Level 3 byte identity is not required. X-09 pins Team A event-envelope v0 and its signal is exactly `buy five seconds after score`; only risk and fill choices remain locked.
- **X-02:** `diff_ms = epoch_ms(timestamp_received) - epoch_ms(timestamp)`. Signed P50/P95/P99 use exact `quantile_cont` linear interpolation over the integer-millisecond frequency distribution; absolute P99 separately applies the same estimator to `abs(diff_ms)`. Hourly drift is UTC-hour-23 median minus UTC-hour-00 median. Disorder partitions by `(market, asset_id)`, canonically sorts by `(timestamp_received, timestamp)`, counts adjacent strict decreases in source `timestamp`, and divides by `n_rows - n_unique_market_asset_streams`. Negative rate `>= 0.001` or absolute P99 `> 5000ms` triggers downgrade from millisecond research to seconds.
- **X-02 / X-03 / X-07:** These are measurement exemptions. A falsified directional hypothesis may still produce a valid completed measurement; it does not become a positive promotion decision.
- **X-04:** Report each ±1/2/5/10/30/60 second tolerance. “Minute-scale researchable, second-scale indeterminate” is a valid conclusion. No unregistered effect threshold is invented.
- **X-06:** Report Brier, log loss, calibration slope and intercept, and game-cluster bootstrap confidence intervals. Gate 1 (Reaction Model) and Gate 2 (trading relevance/cost comparison) are separate; Gate 1 never implies Gate 2.
- **X-08:** Seven days means the continuous overlap of both hash-verified live capture streams under the registered 60-second heartbeat gap policy, not fixture duration or an opaque evidence hash. “No gaps” is the pass criterion; uptime below 99% triggers repair and rerun.
- **X-10:** 45/50 passes the research precision gate; 44/50 does not. A passing artifact must also deliver the relation taxonomy, review-queue design, and G1 go/no-go input. A pass never authorizes live arbitrage.

## 5. Promotion and compliance

Team H can veto promotion. Team I green for the relevant venue, jurisdiction, account type, data license, and operation is required before any real-money action. Until then, every order and fill is simulated or paper-only.

The following remain closed in v0:

1. real-money execution;
2. live maker strategy or trained maker queue point estimate;
3. exact PMXT L2 queue-fill claims;
4. live multi-venue arbitrage;
5. live copy trading;
6. LLM in the hot path;
7. reinforcement learning;
8. large-scale microservice decomposition;
9. unregistered backtests;
10. treating README returns as evidence.

Failure is evidence when it follows the registered design. It is not permission to widen scope.
