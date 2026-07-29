# NFL Factor V4 — Exact-153 Two-Stage Probability Modeling Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to execute this plan task by task.

**Goal:** Select one Stage B probability model on the frozen 153-game
Polymarket development cohort, validate that frozen winner on Kalshi development
data, freeze a cross-venue factor shortlist, and only then evaluate the exact
81-game final holdout once.

**Architecture:** Stage A is the already-published football reference
calibration and is consumed without refitting. Stage B models the historical
market-reaction probability chain on the exact V4/PanelV2 development authority,
publishes 55 walk-forward OOF cells, scores the realized joint outcome with a
proper log score, and applies a frozen one-standard-error rule on Polymarket.
Kalshi is validation only: it receives the frozen Polymarket winner without
target recalibration. The 81-game holdout remains reaction-unread until a
content-addressed shortlist lock exists.

**Tech Stack:** Python 3.12, pandas, NumPy, PyArrow/Parquet, scikit-learn,
XGBoost, pytest, canonical JSON, SHA-256 content-addressed publication.

---

## 1. Scope and claim boundary

This plan governs one two-stage pipeline:

1. **Stage A — existing football reference calibration**
   - Reuse the published 153-game Stage A batch.
   - Consume `p_before_home`, `p_after_home`, reference deltas, and the four
     published calibration tables by verified hash.
   - Do not fit, recalibrate, or republish Stage A during Stage B execution.

2. **Stage B — historical market-reaction probabilities**
   - Fit three conditional probability heads on historical actual-trade data.
   - Select on Polymarket development OOF evidence only.
   - Validate the selected identity on Kalshi development evidence only.
   - Build a factor shortlist from exact shared cross-venue evidence.

The claim boundary is
`HISTORICAL_TRADES_ONLY_SOURCE_TIME_PROBABILITY_DIAGNOSTIC`. The target contract
is `HISTORICAL_TRADES_ONLY_HOME_PROBABILITY`. Historical L2, bid/ask, depth,
queue, fill, continuity, and venue-tick claims remain unsupported. The fixed
direction materiality threshold is `0.01` probability, explicitly a research
threshold rather than a venue tick.

A Polymarket winner may be stated only from a typed
`StageBModelSelectionResult` returned by
`verify_stage_b_v2_selection_result()`. The runner's compact selection JSON is
an audit publication, not that typed result: current code has no public loader
or manifest verifier that reconstructs the result from the JSON. No Kalshi
validation outcome, shortlisted factor, or holdout result may be stated before
its code-authoritative result and required evidence exist and pass their
implemented invariants.

## 2. Frozen authority inputs

All paths are relative to the repository root.

| Authority | Frozen path or identity | Required SHA-256 |
|---|---|---|
| Factor registry | `registries/factors/nfl_factor_registry_v4.json` | file `sha256:92e5001d92afa0748731b5310dae8289ff6930b26a141e981ed910d2c761575f`; semantic `sha256:527a084317ec4a728e5567feea756c1541b65bb814fcf96900b6cfbfd223ead8` |
| Exact-153 V4 facts | `artifacts/market-observation/nfl/x13/exact-153-facts-v4/batches/manifests/sha256/5d/5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757.batch-index.json` | file `sha256:5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757`; batch `sha256:b097f35c30312068ca46e43a0d97e692f30f51a9dcdb89fc1ee604d1be98a082` |
| Exact-153 market batch | `artifacts/market-observation/nfl/x13/factor-lab/v2/expansion-development-market/kalshi-native-time-v3/exact-153/batches/manifests/sha256/b2/b21640b8a50bd92e2f7ed3dac07e641059f8fba9375c1dce0a47a881d655e341.batch-index.json` | file `sha256:b21640b8a50bd92e2f7ed3dac07e641059f8fba9375c1dce0a47a881d655e341` |
| Cohort authority object | `artifacts/market-observation/nfl/x13/factor-lab/v2/expansion-registry/objects/sha256/22/226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d.json` | object `sha256:226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d` |
| Existing Stage A | `artifacts/market-observation/nfl/x15/stage-a-reference-v1/batches/manifests/sha256/50/50040cc83d44f5a62d70cbac92d2aa4d8064bbd4b9c3b36f79fe578bd72a2182.batch-index.json` | file `sha256:50040cc83d44f5a62d70cbac92d2aa4d8064bbd4b9c3b36f79fe578bd72a2182`; batch `sha256:4e3c48222e30589cd932e8ad958aec4bd20e92d2afde7e1f12ca3406f05f8956` |
| Exact-153 PanelV2 | `artifacts/market-observation/nfl/x15/historical-trades-only-development-panel-v2/batches/manifests/sha256/39/39e9f1490a1adcb693c29b9f9fe2f94ec72f1f2d3eafe748ba09e37c7fc750c3.batch-index.json` | file `sha256:39e9f1490a1adcb693c29b9f9fe2f94ec72f1f2d3eafe748ba09e37c7fc750c3`; batch `sha256:d0cb73d381eeb39a7cf5d4cb2ebf24f05037be3e25905ac5e24977c72b3baba8` |
| PanelV2 cohort mapping | PanelV2 batch field | `sha256:f8866ca15ad30f3ab787921aec29db2d700f3df62014de813f5836396d887332` |
| Factor membership rows | Rebuilt only from verified PanelV2 game objects | `sha256:d2fc72c3d81720bcf2bf2a7550272f734544923abbfec72d2165e12cc634a874` |
| Membership artifact bindings | Registry + facts + panel descriptors | `sha256:0725380c27e0353a0f6c92bef482b72b757970981cf6c31102f93fb6c64047c4` |

PanelV2 is `HistoricalTradesOnlyProbabilityPanelV2` and binds exactly:

- 153 development games;
- 25,408 distinct `game_id × atomic_information_episode_id` pairs;
- 83,659 factor-membership rows;
- cohort authority
  `sha256:226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d`;
- `holdout_reaction_accessed=false`.

Any mismatch in path, byte hash, semantic hash, count, schema, cohort authority,
or holdout-access declaration is a hard failure.

## 3. Inputs, outputs, and hash lifecycle

| Step | Verified inputs | Output | Hash rule |
|---|---|---|---|
| Authority assembly | V4 registry, V4 facts, market batch, cohort object, existing Stage A | Exact-153 PanelV2 | Verify the fixed file, batch, semantic, cohort, and mapping hashes above before yielding any partition. |
| Stage B OOF | Verified PanelV2 partitions, frozen folds, 11-cell design matrix | 55 immutable shard manifests plus one selection-ready batch index under `artifacts/market-observation/nfl/x15/stage-b-probability-oof-v2` | Every Parquet object binds byte SHA, schema fingerprint, and semantic-row SHA; every shard and batch is content addressed. |
| Polymarket selection | One verified selection-ready exact-55 batch | One public-verifier-checked in-memory `StageBModelSelectionResult`, followed by one compact audit manifest under `artifacts/market-observation/nfl/x15/stage-b-model-selection-v2` | The typed result binds the complete evidence and hashes. The compact manifest records the input batch hashes, eight run-config SHAs, decision/winner fields, candidate audit records, candidate audit SHA, and winner-rule SHA, but is not self-contained and has no public loader/verifier. |
| Kalshi development validation | The verifier-returned typed `StageBModelSelectionResult`, native/transport Kalshi OOF, frozen cohort metadata | Three exact-pair comparison layers and one development validation result | Kalshi consumes the typed result rather than the compact manifest; bind candidate identity, authority, target/claim contracts, pair grains, training/calibration hashes, and `NO_TARGET_RECALIBRATION`. |
| Factor shortlist | Verified selection, verified Kalshi validation, verified V4/PanelV2 membership | Cross-venue shortlist evidence and `ShortlistLockV1` | Bind exact shared pair identities, support/attrition, BH q-values, leave-one-game-out stability, single-game concentration, registry/facts/panel hashes, and all upstream result hashes. |
| Final holdout | Verified shortlist lock and exact 81-game sealed authority | One-time holdout evaluation artifact | Assign result hashes only after the authorized read and atomic publication; before that point, no result hash or result claim exists. |

## 4. Stage A: consume the existing calibration

Stage A output is immutable input, not work to repeat. Its batch contains exactly
153 game publications and these calibration tables:

| Table | Object SHA-256 | Semantic rows SHA-256 |
|---|---|---|
| `calibration_summary` | `sha256:09e1db48377a8437eb646b5d8e176b44c72c854338f61e3610bb1349019e886e` | `sha256:edb1e0c84beaf435d77e7b665bcad45175de81d4204302928fc094aa9f4fcabc` |
| `calibration_reliability` | `sha256:bc833f156a230754680d732b30fe39f4eed68f468eec2a03e6dc7c4103e50516` | `sha256:0d304e91d95b419fb7f6a1ef746df04af50b18817f3e98b971851cd3126e1043` |
| `calibration_breakdowns` | `sha256:178fdfc88f5f1539dbcd73028126f0beb4defc6337709bddfa9170808541a73b` | `sha256:18ed85a7fce391ef2a7f5837b2d132c18050b5d6862fc7144e2f2161aa99d290` |
| `calibration_bootstrap` | `sha256:dbecd7ce82c8b727217752fa37efa5234883956e161005fce11ec93e8fac5d1c` | `sha256:24f6a75b9216e32784b19dd4287e3cd656cf292211649206c8cda93918b26098` |

The calibration uses equal total game weight and a fixed game-cluster bootstrap.
PanelV2 must verify the Stage A batch file hash before attaching reference
probabilities. Stage B may use Stage A columns only through the frozen `D4`
feature block.

## 5. Stage B probability and score contract

For landmark `L` and endpoint `H`, Stage B represents the realized path as:

1. `S_H`: endpoint survival/observability;
2. `O_H | S_H`: a fresh actual trade at the surviving endpoint;
3. `D_H | S_H, O_H`: `DOWN`, `NO_MOVE`, or `UP`.

The survival contract is `DISCRETE_INTERVAL_SURVIVAL_PRODUCT_V1`. Ordered
interval probabilities are multiplied along the at-risk path:

```text
P(S_H = 1 | x_L)
  = product over intervals (a, b] from L through H
    P(S_b = 1 | S_a = 1, x_L)
```

Each binary head contributes Bernoulli negative log likelihood when its truth is
defined. The direction head contributes categorical negative log likelihood
when its truth is defined. The proper realized-path score is:

```text
joint_row_nll
  = nll(S_H)
  + nll(O_H | S_H), when defined
  + nll(D_H | S_H, O_H), when defined
```

This is the negative log of the applicable conditional-chain probability. It is
the sum of defined conditional terms and is never divided by the number of
defined terms. Candidate and B0 must have symmetric head availability.

The hierarchy is fixed:

1. pair candidate and B0 on exact source row, game, episode, venue, fold,
   landmark, endpoint, authority, target, and claim fields;
2. average rows within `game_id × episode × venue`;
3. average episodes within game;
4. give every game equal weight;
5. bootstrap paired game improvements 10,000 times with seed `20260729`.

Positive improvement means candidate joint loss is lower than B0. The integrated
gate requires positive mean and positive 95% paired-game-bootstrap lower bound.
The clean anchor is fixed at `L=3`, `H=30`, requires at least 30 games, and must
not reverse the integrated sign.

## 6. Frozen Stage B experiment matrix

### Controls

1. `b0_empirical_v1 / D0`
2. `regularized_logistic_v1 / D0`
3. `shallow_xgboost_v1 / D0`

### Selectable candidates

1. `regularized_logistic_v1 / D1`
2. `regularized_logistic_v1 / D2`
3. `regularized_logistic_v1 / D3`
4. `regularized_logistic_v1 / D4`
5. `shallow_xgboost_v1 / D1`
6. `shallow_xgboost_v1 / D2`
7. `shallow_xgboost_v1 / D3`
8. `shallow_xgboost_v1 / D4`

### Expanding time folds

| Fold | Train weeks | Validation weeks |
|---|---|---|
| `fold_01` | 1–2 | 3–4 |
| `fold_02` | 1–4 | 5–6 |
| `fold_03` | 1–6 | 7–8 |
| `fold_04` | 1–8 | 9–10 |
| `fold_05` | 1–10 | 11–12 |

The publication requirement is exactly `11 design cells × 5 folds = 55` unique
shards. A partial batch is diagnostic only and cannot enter selection.

All fitting, preprocessing, and calibration occur inside the training side of
each fold. Polymarket selection must never consume Kalshi metrics. Controls
cannot win. Only the eight selectable candidates enter the winner decision.

## 7. Polymarket one-standard-error selection

For each candidate:

1. verify the exact B0 pairing and frozen contracts;
2. compute the proper joint loss improvement at row, episode, and game grain;
3. apply authority, integrated, and clean-anchor gates;
4. compute the standard error of the per-game improvements.

Among gate-passing candidates, find the highest mean improvement and set:

```text
one_se_threshold = best_mean_improvement - best_standard_error
```

Choose the first candidate in the frozen suite order whose mean improvement is
at least that threshold. If no candidate passes all gates, publish
`NO_MODEL_ADVANCE`; do not manufacture a winner.

The in-memory `StageBModelSelectionResult` contains:

- `paired_rows`;
- `episode_losses`;
- `game_losses`;
- `anchor_game_losses`;
- the complete model-selection result;
- the complete eight-candidate audit.

Each DataFrame commitment uses
`nfl_x15_typed_dataframe_evidence_v2`: typed row and column Index metadata,
typed dtypes, typed values, ordered records, and schema identity are all hashed.
`verify_stage_b_v2_selection_result()` accepts this typed result and rejects
dtype drift, index drift, row reordering, payload mutation, duplicate identity,
or hash mismatch.

After that verifier returns, `scripts/research/select_nfl_x15_stage_b_v2.py`
publishes a compact JSON containing the decision status, optional winner,
candidate audit records, upstream batch/run-config hashes, audit hash, and
winner-rule hash. It does not serialize the four evidence DataFrames or the
complete typed result. The JSON therefore cannot be described as
self-verifying, cannot be loaded back into `StageBModelSelectionResult` by
current code, and cannot replace the typed object at the Kalshi boundary.

## 8. Polymarket selection to Kalshi validation

Kalshi validation starts only from a typed `StageBModelSelectionResult` returned
by `verify_stage_b_v2_selection_result()` with
`decision_status="MODEL_ADVANCE"` and a non-null Polymarket winner. The compact
selection manifest is not an accepted input. The model ID, feature block,
authority, folds, target contract, claim boundary, and score contract remain
frozen.

The selection and Kalshi steps must retain or rebuild the typed result from the
verified exact-55 inputs within the controlled execution path. If only the
compact manifest remains after a process boundary, current code cannot resume
Kalshi validation from it; stop rather than inventing a manifest verification
path.

Run the winner and B0 on Kalshi development data in two modes:

- **transported:** fitted/calibrated from Polymarket training data and applied to
  Kalshi, with `NO_TARGET_RECALIBRATION`;
- **native:** fitted/calibrated from Kalshi training data for a venue-native
  comparison.

Publish these exact comparison layers:

1. transported candidate versus native Kalshi candidate;
2. transported candidate versus transported B0;
3. native Kalshi candidate versus native Kalshi B0.

Every layer must use identical Kalshi truth rows, exact pair identity, and the
same proper joint score. The transported candidate-versus-transported-B0 layer
is the Kalshi factor-effect input. Kalshi may validate or reject transport; it
may not replace the Polymarket selection decision.

## 9. Cross-venue factor shortlist

Build factor membership only from the verified V4 registry, V4 facts, and
PanelV2 authority. Join both venues at:

```text
game_id
× nfl_week
× atomic_information_episode_id
× factor_id
× factor_version
× landmark_seconds
× endpoint_seconds
× fold_id
× candidate model/block
× baseline model/block
```

Use only the clean `L=3 → H=30` candidate-versus-B0 evidence shared by both
venues. Preserve and publish venue-only attrition rather than silently dropping
it.

A factor enters the shortlist only when all of these are true:

- at least 30 shared games and 20 shared episodes;
- both venues have supported factor evidence;
- both venue means have confidence intervals excluding zero;
- both Benjamini–Hochberg q-values are at most `0.05`;
- both leave-one-game-out same-sign rates are at least `0.80`;
- both maximum single-game absolute contribution ratios are at most `0.25`;
- both venue development gates pass;
- cross-venue signs agree;
- the global transport gate passes;
- the upstream Polymarket selection gate passes.

Freeze the resulting identities, versions, gates, statistics, exact pair hashes,
attrition hashes, registry/facts/panel hashes, selection hash, Kalshi validation
hash, code hash, and holdout policy into `ShortlistLockV1`. An empty shortlist is
a valid frozen outcome and does not authorize inventing replacement factors.

## 10. Shortlist lock to exact-81 final holdout

Before the shortlist lock:

- all 81 holdout rows must declare
  `market_reaction_exposure=SEALED_UNREAD`;
- total holdout `reaction_read_count` must equal `0`;
- only cohort metadata may be verified;
- no holdout prediction, metric, factor ranking, or reaction summary may be
  computed.

After the lock passes its public verifier:

1. authorize one exact read of the 81 frozen holdout games;
2. evaluate only the frozen winner, B0, shortlist identities, factor versions,
   landmarks, endpoints, thresholds, score, and metrics;
3. prohibit refitting, target recalibration, threshold changes, factor
   substitution, or a second selection pass;
4. publish one atomic content-addressed holdout artifact with the shortlist-lock
   SHA, exact 81-game authority SHA, read ledger, object byte hashes, semantic
   hashes, code hash, and all reported metrics;
5. close the read ledger and report the result exactly as published, including
   failure or an empty evaluable set.

The current development-only validation module intentionally has no holdout
reaction-read API. The holdout executor must be a separate lock-gated entry
point so development code cannot open the sealed cohort.

## 11. Current status as of 2026-07-29

### Completed and hash-verifiable

- [x] V4 factor registry and exact-153 V4 facts authority are published.
- [x] The existing 153-game Stage A batch and four calibration tables are
  published.
- [x] Exact-153 `HistoricalTradesOnlyProbabilityPanelV2` is published and binds
  the frozen authorities and closed holdout declaration.
- [x] Stage B survival, fitting, fold-local calibration, exact pairing,
  proper-joint-score, hierarchical weighting, bootstrap, clean-anchor gate, and
  one-standard-error selection contracts are implemented.
- [x] The 3-control/8-candidate design matrix, five folds, resumable shard
  publication, and exact-55 selection-readiness gate are implemented.
- [x] Typed DataFrame evidence V2 and the public selection verifier are
  implemented.
- [x] Polymarket-to-Kalshi development validation, exact comparison layers,
  frozen membership verification, and cross-venue shortlist gates are
  implemented and test-covered.

### Currently executing

- [ ] Produce the exact-55 Stage B OOF batch under
  `artifacts/market-observation/nfl/x15/stage-b-probability-oof-v2` with
  `scripts/research/run_nfl_x15_exact153_oof.py`.

At this snapshot, no selection-ready V2 batch hash is recorded in this plan, no
typed Stage B selection result has been produced, and no compact Stage B
selection manifest exists. The execution may publish only progress artifacts
until all 55 cells verify. This section records pipeline state, not a model
result.

### Blocked on upstream artifacts, not yet executed

- [ ] Run and publish Polymarket one-standard-error selection from the verified
  exact-55 batch.
- [ ] Run and publish Kalshi development validation for the verified winner.
- [ ] Build and freeze the cross-venue factor shortlist.
- [ ] Open and evaluate the exact-81 holdout once after shortlist-lock
  verification.

No downstream result is implied by implemented code or passing unit tests.

## 12. Execution tasks and verification

### Task 1: Verify frozen authorities before resuming OOF

**Files:**

- Verify: `src/prediction_market/research/nfl_x15_development_panel.py`
- Verify: `tests/research/test_nfl_x15_development_panel.py`
- Verify: the exact authority paths in Section 2

- [ ] Run:

```bash
pytest -q tests/research/test_nfl_x15_development_panel.py
```

- [ ] Verify the PanelV2 batch file hash, batch hash, 153 game count, cohort
  authority, cohort mapping, and `holdout_reaction_accessed=false`.
- [ ] Stop on any mismatch; do not regenerate an authority in place.

### Task 2: Complete and verify the exact-55 OOF batch

**Files:**

- Execute: `scripts/research/run_nfl_x15_exact153_oof.py`
- Verify: `src/prediction_market/research/nfl_x15_oof_publication.py`
- Test: `tests/research/test_nfl_x15_oof_publication.py`
- Output: `artifacts/market-observation/nfl/x15/stage-b-probability-oof-v2`

- [ ] Resume the runner against the fixed PanelV2 manifest and its exact file
  SHA.
- [ ] Require exactly 55 unique fold/model/block shard manifests.
- [ ] Require the batch publication status to be selection-ready.
- [ ] Record the final batch manifest path, file SHA, and batch SHA only after
  public verification.
- [ ] Run:

```bash
pytest -q \
  tests/research/test_nfl_x15_models.py \
  tests/research/test_nfl_x15_oof_publication.py
```

### Task 3: Publish the Polymarket selection decision

**Files:**

- Execute: `scripts/research/select_nfl_x15_stage_b_v2.py`
- Verify: `src/prediction_market/research/nfl_x15_model_selection.py`
- Test: `tests/research/test_nfl_x15_model_selection.py`
- Output: `artifacts/market-observation/nfl/x15/stage-b-model-selection-v2`

- [ ] Pass the verified exact-55 batch manifest and its observed file SHA.
- [ ] Verify all eight candidate projections share the frozen authority and run
  contract.
- [ ] Apply the authority, integrated, clean-anchor, and one-standard-error
  rules without reading Kalshi selection metrics.
- [ ] Call `verify_stage_b_v2_selection_result()` on the typed
  `StageBModelSelectionResult`.
- [ ] Publish the compact audit manifest with either `MODEL_ADVANCE` and its
  winner fields or `NO_MODEL_ADVANCE` and a null winner.
- [ ] Do not claim the compact manifest is self-contained or publicly loadable.
- [ ] Run:

```bash
pytest -q tests/research/test_nfl_x15_model_selection.py
```

### Task 4: Validate the frozen winner on Kalshi development data

**Files:**

- Verify: `src/prediction_market/research/nfl_x16_kalshi_validation.py`
- Test: `tests/research/test_nfl_x16_kalshi_validation.py`

- [ ] Pass the typed `StageBModelSelectionResult` to
  `verify_stage_b_v2_selection_result()` and then into Kalshi validation; do not
  pass or reload the compact manifest.
- [ ] If the decision is `NO_MODEL_ADVANCE`, publish no Kalshi winner claim and
  do not continue to factor shortlisting.
- [ ] Otherwise build transported and native Kalshi development OOF evidence
  without target recalibration.
- [ ] Publish and verify all three exact comparison layers.
- [ ] Run:

```bash
pytest -q tests/research/test_nfl_x16_kalshi_validation.py
```

### Task 5: Freeze the factor shortlist

**Files:**

- Execute: `src/prediction_market/research/nfl_x16_kalshi_validation.py`
- Reuse lock schema: `src/prediction_market/sports/nfl_factor_lab_analysis.py`
- Test: `tests/research/test_nfl_x16_kalshi_validation.py`

- [ ] Rebuild and hash factor membership from the frozen artifact authorities.
- [ ] Construct exact shared cross-venue clean-anchor pairs.
- [ ] Publish support, venue-only attrition, BH, leave-one-game-out, and
  concentration evidence.
- [ ] Freeze and publicly verify `ShortlistLockV1`.
- [ ] Confirm the holdout reaction read count is still zero.

### Task 6: Perform the one-time exact-81 holdout evaluation

**Files:**

- Add: `src/prediction_market/research/nfl_x17_holdout_evaluation.py`
- Add: `scripts/research/run_nfl_x17_exact81_holdout.py`
- Add: `tests/research/test_nfl_x17_holdout_evaluation.py`
- Output: `artifacts/market-observation/nfl/x17/exact-81-holdout-v1`

- [ ] Make verified `ShortlistLockV1` and the exact sealed 81-game authority
  mandatory constructor inputs.
- [ ] Open only the 81 reaction objects named by that authority.
- [ ] Evaluate only frozen identities and metrics.
- [ ] Atomically publish the result and closed read ledger.
- [ ] Prove a second invocation cannot select, refit, or alter the frozen
  specification.
- [ ] Run:

```bash
pytest -q tests/research/test_nfl_x17_holdout_evaluation.py
```

### Task 7: End-to-end claim audit

- [ ] Run:

```bash
pytest -q \
  tests/research/test_nfl_stage_a_reference.py \
  tests/research/test_nfl_x15_development_panel.py \
  tests/research/test_nfl_x15_models.py \
  tests/research/test_nfl_x15_oof_publication.py \
  tests/research/test_nfl_x15_model_selection.py \
  tests/research/test_nfl_x16_kalshi_validation.py \
  tests/research/test_nfl_x17_holdout_evaluation.py
```

- [ ] Verify every reported input and output hash against bytes on disk.
- [ ] Verify Polymarket was the only selection venue and Kalshi was validation
  only.
- [ ] Verify the shortlist lock predates the first holdout reaction read.
- [ ] Verify every holdout statement is derived from the single published
  exact-81 artifact.
- [ ] Report missing, rejected, empty, or failed evidence as such; never convert
  it into a success claim.
