# NFL Factor V4 — Exact-153 Two-Stage Probability Modeling Plan

> **Execution rule:** use the existing immutable 153-game artifacts. Do not
> recapture, replay, or reinterpret a published single-game bundle.

## 1. Goal and claim boundary

This phase answers two separate questions:

1. **Stage A — football reference value**
   - Given the completed event and resulting NFL state, how much did the
     diagnostic football win probability change?
2. **Stage B — market-reaction probability**
   - At decision landmark `L`, what is the probability that an actual trade is
     observable by endpoint `H`, and if observable, is the price `DOWN`,
     `NO_MOVE`, or `UP`?

The sequence is fixed:

```text
153-game canonical facts
→ existing Stage A reference
→ exact-45 Stage B OOF
→ Polymarket-only model selection
→ frozen-winner Kalshi validation
→ factor predictive-utility review
→ shortlist lock
→ one-time 81-game holdout
```

All Stage B results remain:

```text
HISTORICAL_TRADES_ONLY_SOURCE_TIME_PROBABILITY_DIAGNOSTIC
```

They are not evidence of live latency, executable bid/ask, fill, causality, or
tradable alpha. Historical L2, OFI, depth, queue, maker fill, and midpoint P&L
remain outside this experiment.

## 2. Frozen inputs

All paths are relative to the repository root.

| Authority | Path | Binding |
|---|---|---|
| Factor registry V4 | `registries/factors/nfl_factor_registry_v4.json` | file `sha256:92e5001d92afa0748731b5310dae8289ff6930b26a141e981ed910d2c761575f` |
| Exact-153 facts V4 | `artifacts/market-observation/nfl/x13/exact-153-facts-v4/batches/manifests/sha256/5d/5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757.batch-index.json` | 153 games; 26,192 reconciled rows; 83,659 factor hits; zero silent loss |
| Exact-153 market batch | `artifacts/market-observation/nfl/x13/factor-lab/v2/expansion-development-market/kalshi-native-time-v3/exact-153/batches/manifests/sha256/b2/b21640b8a50bd92e2f7ed3dac07e641059f8fba9375c1dce0a47a881d655e341.batch-index.json` | Polymarket and Kalshi historical actual trades |
| Cohort authority | `artifacts/market-observation/nfl/x13/factor-lab/v2/expansion-registry/objects/sha256/22/226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d.json` | exact game/week/kickoff/batch identity |
| Stage A reference | `artifacts/market-observation/nfl/x15/stage-a-reference-v1/batches/manifests/sha256/50/50040cc83d44f5a62d70cbac92d2aa4d8064bbd4b9c3b36f79fe578bd72a2182.batch-index.json` | immutable retrospective reference |
| PanelV2 | `artifacts/market-observation/nfl/x15/historical-trades-only-development-panel-v2/batches/manifests/sha256/39/39e9f1490a1adcb693c29b9f9fe2f94ec72f1f2d3eafe748ba09e37c7fc750c3.batch-index.json` | file `sha256:39e9f1490a1adcb693c29b9f9fe2f94ec72f1f2d3eafe748ba09e37c7fc750c3` |
| Stage A decision-time support audit | `artifacts/market-observation/nfl/x15/stage-a-decision-support-audit-v1/manifests/sha256/17/17593d410b93a6eb8d21f2deedd8d78d1b566bb9ec41e51a4870ec4acedee219.stage-a-support-audit.json` | file `sha256:17593d410b93a6eb8d21f2deedd8d78d1b566bb9ec41e51a4870ec4acedee219` |

PanelV2 binds:

- 153 development games;
- 2,876,562 panel rows;
- 1,050,084 decision rows;
- 25,408 distinct game × episode pairs;
- 83,659 factor memberships;
- Polymarket and Kalshi rows at identical panel grain;
- `holdout_reaction_accessed=false`.

Any path, byte hash, semantic hash, schema, authority, mapping, or holdout
declaration mismatch fails closed.

## 3. Stage A status and the D4 decision

Stage A is already published. Its development calibration is:

| Metric | Value |
|---|---:|
| Brier | 0.158593 |
| Brier game-bootstrap 95% CI | 0.138409–0.179448 |
| Log loss | 0.470834 |
| Log-loss game-bootstrap 95% CI | 0.419756–0.523813 |
| Calibration slope | 1.091129 |
| Calibration intercept | -0.021436 |

It is a retrospective diagnostic reference, not a live truth source.

The Stage A support audit established:

- `p_before_home`, `p_after_home`, `reference_delta_home`, and
  `reference_gap_at_landmark` are non-null on **zero** of the 1,050,084 Stage B
  decision rows;
- the post-event state becomes known after the early `L=1/2/3/5/10` decision
  landmarks;
- therefore D4 would leak unavailable information and is
  `UNSUPPORTED_FEATURE_BLOCK`.

D4 is excluded from selection and execution. The one already-published D4
shard remains immutable diagnostic evidence but is ignored.

## 4. Stage B target and time grid

The frozen landmarks and endpoints are:

```text
L ∈ {1, 2, 3, 5, 10} seconds
H ∈ {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60} seconds
H > L
```

At each `L → H`, Stage B models a conditional chain:

1. `S_H`: the path survives without a censoring event;
2. `O_H | S_H`: a fresh actual trade is observable at the endpoint;
3. `D_H | S_H, O_H`: `DOWN`, `NO_MOVE`, or `UP`.

The proper joint row score is:

```text
NLL(S_H)
+ I(S_H) × NLL(O_H | S_H)
+ I(S_H, O_H) × NLL(D_H | S_H, O_H)
```

The direction term is also evaluated independently with categorical log loss
and three-class Brier score. Missing endpoint trades are not forward-filled.

## 5. Feature blocks and models

| Block | Inputs | Role |
|---|---|---|
| D0 | landmark and endpoint only | empirical/model controls |
| D1 | D0 + pre-landmark price, staleness, prior 30/60s trade count and size | market-state candidate |
| D2 | D1 + clock, score, possession, down, distance, field position | game-state candidate |
| D3 | D2 + primary action, yards/returns, actor/beneficiary orientation, Factor V4 multi-hot membership | event-factor candidate |
| D4 | D3 + Stage A before/after reference fields | prohibited: unavailable at decision time |

Models:

- `b0_empirical_v1`;
- `regularized_logistic_v1`;
- `shallow_xgboost_v1`.

Preprocessing, fitting, and calibration happen only inside each training fold.
No target-venue recalibration occurs during Kalshi transport validation.

## 6. Expanding chronological folds

| Fold | Train weeks | Validation weeks |
|---|---|---|
| fold_01 | 1–2 | 3–4 |
| fold_02 | 1–4 | 5–6 |
| fold_03 | 1–6 | 7–8 |
| fold_04 | 1–8 | 9–10 |
| fold_05 | 1–10 | 11–12 |

The frozen exact-45 execution matrix is:

- 5 × `b0_empirical_v1 / D0`;
- 10 × logistic/XGBoost D0 controls;
- 30 × logistic/XGBoost × D1/D2/D3 selectable candidates.

Current reusable state:

- 6 supported immutable V4 shards already exist;
- 1 D4 shard is verified then ignored;
- 39 supported cells remain to compute;
- partial batches are diagnostic only;
- selection requires all 45 cells.

The runner prepares PanelV2 once, reuses the in-memory prepared frame, publishes
each completed cell atomically, and resumes only after full shard verification.
Only one model process may run on the 16-GiB host.

## 7. Exact-45 verification and provenance

Every V4 shard is verified for:

- exact namespace and fold/model/block coordinate;
- canonical manifest semantic hash;
- byte SHA and length for every Parquet object;
- Parquet row count, schema fingerprint, and columns;
- experiment, cohort, target, claim, selection-role, magnitude, authority, and
  holdout contracts;
- fold train/validation/preprocessor identities;
- effective seed and shared run-config contracts.

Batch verification is memory-bounded:

- no full pandas load for the 45-cell batch;
- provenance columns are scanned with PyArrow
  `iter_batches(batch_size=1024)`;
- the batch stores a canonical typed provenance anchor, one row per cell;
- a selection projection fully loads only B0 plus one candidate across five
  folds: exactly 10 shards;
- those 10 loaded rows are rederived and compared with the sealed anchor.

The provenance anchor binds model-spec, training, preprocessing, calibration,
fold, source-run, OOF-object, shard, and manifest hashes. A result cannot
self-declare its authority.

## 8. Polymarket selection

Polymarket is the only selection venue. Each candidate is compared with B0 on
exact paired rows.

Weighting:

1. equal rows within episode;
2. equal episodes within game;
3. equal games in the final estimate.

Inference:

- 10,000 paired game-cluster bootstrap draws;
- seed `20260729`;
- integrated joint-loss improvement mean and 95% CI;
- integrated direction log-loss improvement mean and 95% CI;
- integrated direction Brier improvement mean and 95% CI;
- clean anchor `L=3 → H=30`, with at least 30 games;
- anchor joint and direction means cannot reverse sign.

A candidate passes only if all joint and direction gates pass. One-standard-
error selection runs only among passing candidates. Controls cannot win. If
none pass, the result is `NO_MODEL_ADVANCE`.

The verifier independently rederives:

- row losses;
- episode/game losses;
- bootstrap distributions and CIs;
- anchor evidence;
- gates, selection status, and one-SE decision;
- every game/week/fold and train/validation/preprocessor identity;
- the independent exact-45 authority and provenance binding.

## 9. Kalshi development validation

Kalshi never selects, reranks, or recalibrates the Polymarket winner.

The validator internally reloads the frozen winner from the exact-45
content-addressed batch and requires:

- identical source row, contract, pair, truth, and probability identities;
- exact fold, model-spec, training, preprocessing, and calibration provenance;
- identical Kalshi truth populations across transported and native layers;
- transported winner versus transported B0 joint-loss gate;
- independent `S_H` and `O_H | S_H` nuisance-head gates;
- direction log-loss and Brier gates;
- `L=3 → H=30` minimum 30-game anchor;
- native Kalshi headwise non-inferiority as a hard final gate.

Native Kalshi is a comparator, not a second model-selection venue. Any transport
failure rejects cross-venue promotion and does not return to Polymarket to pick
a different model.

## 10. Factor predictive utility and shortlist

Factor-conditioned output is named **predictive utility**:

```text
candidate-vs-B0 predictive-loss improvement
conditioned on Factor V4 membership
```

It is not `gross_markout`, factor effect, signed reaction, signal, alpha, or
tradable return.

Frozen support floors:

- at least 30 games;
- at least 20 episodes;
- caller overrides below those floors are rejected.

Shortlist creation requires:

- verified Polymarket winner;
- verified Kalshi development validation;
- identical cross-venue pair population;
- current, recomputed authority metadata;
- a pre-holdout lock that still matches current metadata;
- holdout reaction read count exactly zero.

An old lock cannot authorize a shortlist after any holdout metadata mutation.

## 11. Conditional magnitude follow-up

Only after the probability winner is frozen:

- rerun the winner feature block and D0 on the same five folds and exact pairs;
- condition on realized `UP` and `DOWN`;
- estimate `|Δp|` quantiles `q10/q25/q50/q75/q90`;
- compare pinball loss, approximate CRPS, interval coverage, clean anchor, and
  paired game-bootstrap stability.

Magnitude failure blocks magnitude and signed-distribution claims only. It does
not invalidate an otherwise verified direction-probability winner.

## 12. Holdout

The 81-game final holdout remains sealed until:

1. all 45 Stage B cells verify;
2. a Polymarket winner exists;
3. Kalshi validates that frozen winner;
4. factor predictive-utility review is complete;
5. a content-addressed shortlist lock verifies current holdout metadata.

Then and only then:

- authorize one read;
- run the frozen model/factors without changing feature blocks, horizons,
  thresholds, or gates;
- publish one atomic result;
- never tune and reread the same holdout.

Until that point:

```text
holdout_reaction_accessed = false
reaction_read_count = 0
```

## 13. Execution status

Completed:

- [x] Factor V4 exact-153 facts and coverage.
- [x] Stage A reference and calibration.
- [x] Decision-time support audit; D4 excluded.
- [x] Exact-45 artifact/runner contract.
- [x] Full statistical rederivation and independent authority verifier.
- [x] Kalshi frozen-winner validation and current-metadata lock contract.
- [x] Combined focused regression: 193 tests passed.

Next:

- [ ] Run the 39 missing Stage B cells in one resumable process.
- [ ] Publish and verify the exact-45 selection-ready batch.
- [ ] Run Polymarket selection once.
- [ ] If a winner exists, run Kalshi development validation.
- [ ] Publish factor predictive-utility review.
- [ ] Freeze shortlist.
- [ ] Run conditional magnitude follow-up.
- [ ] Read the 81-game holdout exactly once.
- [ ] Publish the final Chinese report and Master Notebook update.

## 14. NO-GO

- No real money or orders.
- No maker, hedging, or cross-venue execution.
- No historical L2/BBO fabrication.
- No queue/fill claim from L2.
- No midpoint treated as executable P&L.
- No holdout read before lock.
- No unregistered factor or post-result horizon tuning.
- No README/leaderboard return claim as evidence.
- No LLM hot path, RL, or microservice expansion.
