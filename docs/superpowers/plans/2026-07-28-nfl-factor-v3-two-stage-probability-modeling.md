# NFL Factor V3 — C1.1 Exact-153 Two-Stage Implementation Plan

> Status: approved unanimously on 2026-07-29 by Sports Scientist, Quant Analyst,
> and Adversarial Reviewer. Execute with `superpowers:subagent-driven-development`.

## 1. Goal

Reuse the verified 153-game development corpus to build two distinct models:

1. **Stage A — football reference value**
   - Apply the frozen no-spread fastrmodels XGBoost model.
   - Produce auditable home-win probabilities before and after eligible atomic
     information episodes.
   - Do not retrain or recalibrate the football model on these 153 games.

2. **Stage B — observed market reaction**
   - Predict whether a clean future endpoint remains observable.
   - Conditional on a fresh actual trade, predict direction and magnitude of the
     home-outcome price change from decision landmark `L` to endpoint `H`.
   - Train and evaluate Polymarket and Kalshi separately, then run exact-pair and
     temporal transport diagnostics.

The 81-game final cohort remains unread until every fact, feature, target,
threshold, model, calibrator, exclusion, and metric is frozen in
`ShortlistLockV1`.

## 2. Inputs That Must Be Reused

- 153 verified X-13 development game bundles.
- 4,773,299 historical actual-trade observations.
- 610,932 historical reaction paths.
- Frozen nflverse play-by-play and participation objects.
- Frozen no-spread fastrmodels XGBoost asset and SHA-256.
- Existing X-15 landmark, walk-forward, publication, and S3 hydrate machinery.
- Existing content-addressed manifests and hashes.

Do not:

- redownload or recapture the 153 games;
- replay upstream market APIs;
- edit old immutable artifacts or manifests;
- read holdout reaction data before the lock;
- forward-fill missing trades;
- infer historical L2, bid/ask, OFI, depth, queue, or fill;
- train on derived home complements as if they were actual home trades;
- count multiple factor tags as multiple training observations.

DAL–DET remains a semantic audit fixture. Its validated schema is expanded to the
frozen 153-game development cohort; it is not silently inserted into that cohort.

## 3. Resource Budget

The host has 16 GiB RAM and showed 13.22 GiB of swap already in use. Current
memory pressure is healthy, but the implementation must avoid a second spike.

- Default heavy batch workers: `1`.
- Process one game at a time.
- Write per-game Parquet/JSON artifacts before releasing DataFrames.
- Do not hold the full 153-game sports and market panels in pandas simultaneously.
- Aggregate through DuckDB/Arrow scans of published per-game objects.
- Model training must set explicit thread limits.
- Increase worker count only after measured peak RSS leaves at least 4 GiB headroom.

## 4. Canonical Data Model

### 4.1 Sports fact grain

`EpisodeFactV3` grain:

```text
game_id × atomic_information_episode_id
```

Required identity and timing:

```text
game_id
raw_play_id
atomic_information_episode_id
score_sequence_id
adjudication_sequence_id
source_interval_start
source_interval_end
source_resolution
information_status       PROVISIONAL | FINAL | REVERSED
stage_b_information_event_eligible
final_sports_outcome_eligible
known_at
source_hashes
```

Required orientation:

```text
home_team
away_team
actor_team
beneficiary_team
actor_is_home
beneficiary_is_home
possession_is_home
beneficiary_resolution_status
```

`beneficiary_team` may only be derived from finalized sports rules. It may not be
derived from market movement, reference sign, or final game result. Unresolvable
beneficiary remains null/`UNRESOLVED`.

Required state and transition facts include:

- period, exact game seconds remaining, score, signed home margin;
- possession, offense, defense, down, distance, yardline, red zone, goal-to-go;
- timeouts and drive identity;
- pass, run, scramble, sack, punt, kickoff, FG, try, kneel, spike;
- completion, pass depth/location, air yards, YAC;
- run location/gap, yards, return yards, kick distance;
- first down, third/fourth conversion/failure;
- INT, fumble, lost fumble, recovery, turnover on downs, muff;
- passing/rushing/defensive/return TD, FG, PAT, 2PT, defensive 2PT, safety;
- touchback, fair catch, inside-20, onside, blocked kick;
- penalty disposition, review, reversal, timeout, no-play, deleted/admin;
- stable player IDs and explicit source-supported roles;
- participation quality and explicit source-supported injury/return evidence.

Pregame injury timeline, official substitution timestamp, routes, coverage,
tracking, pressure assignment, public betting percentage, and PFF grade remain
`DATA_GAP`.

### 4.2 Market reaction grain

`VenueReactionPanelV3` grain:

```text
game_id
× atomic_information_episode_id
× venue
× actual_home_outcome_contract
× decision_landmark_L
× endpoint_H
```

Rules:

- Factor and event tags are multi-hot columns on this row.
- `factor_id` is never part of the training primary key.
- H-missing rows remain in the availability panel.
- Actual home trades and away-derived home complements are separate records.
- Only actual home-outcome trades enter primary training targets.
- Derived complements remain non-observed and non-executable diagnostics.

## 5. Atomic Event and Adjudication Rules

- TD, PAT, and 2PT are separate atomic information episodes.
- `score_sequence_id` connects TD to its try sequence.
- Pick-six is one atomic event with both turnover and defensive-TD tags.
- A retried snap is a new atomic action.
- A timestamped provisional ruling, review initiation, review decision, and
  reversal can each be a Stage B information event.
- `adjudication_sequence_id` connects those information events.
- The next independent information event censors the previous reaction window.
- Only a final effective sports result enters the stat ledger and Stage A state
  transition.
- Nullified/reversed results do not enter final score, turnover, or player stats.
- If the historical source lacks an independent revision timestamp, do not invent
  a provisional/reversal sequence; retain only provable finalized information.
- Atomic-event and sequence-level aggregates may both be published, but they may
  not both count as independent samples in one estimate.

## 6. Stage A — Frozen Football Reference

### 6.1 Model

Use the existing version-pinned no-spread fastrmodels XGBoost model:

```text
model asset SHA-256: frozen existing registry value
runtime: xgboost 3.3
orientation: home-win probability
training in this phase: none
recalibration in this phase: none
formal support: regulation
OT: MODEL_SUPPORT_UNPROVEN
```

Every model input must publish:

```text
feature_name
feature_value
feature_known_at
source_row_id
source_hash
PIT_status
```

`receive_2h_ko` is fail-closed:

- use only evidence known before the decision state;
- never fill first-half states by looking at the Q3 receiving team;
- if no auditable as-of source exists, the reference row is unsupported.

### 6.2 Pre/post purity

```text
p_before_home = P(home wins | pure state before episode)
p_after_home  = P(home wins | pure state after episode)
reference_delta_home = p_after_home - p_before_home
```

An atomic reference delta is eligible only when no independent PAT, 2PT, kickoff,
return, review decision, or next snap lies between the pre/post model states.

Otherwise publish:

```text
reference_status = COMPOSITE_TRANSITION
intervening_episode_ids
bridge_delta_home       optional diagnostic only
```

Composite bridge deltas must not be presented as the atomic event effect.

### 6.3 Output and evaluation

`ReferenceValueObservationV1` contains:

```text
game_id
episode_id
p_before_home
p_after_home
reference_delta_home
reference_status
pre/post source rows and timestamps
model/input hashes
claim_boundary = RETROSPECTIVE_DIAGNOSTIC
```

Evaluate supported regulation states against final game outcome with:

- equal total weight per game;
- Brier score and log loss;
- calibration slope/intercept and reliability table;
- quarter/time/probability breakdown;
- game-cluster bootstrap confidence intervals.

Tied final games are excluded from binary scoring. These metrics evaluate overall
state-probability calibration; they do not make each event delta ground truth.

## 7. Stage B — Observability, Direction, and Distribution

### 7.1 Time grid

Event anchors use `[source_interval_start, source_interval_end)`.

```text
L = 1, 2, 3, 5, 10 seconds after source_interval_end
H = 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 seconds
H > L
```

- Same-second or overlapping event/market intervals are `order_ambiguous`.
- Historical source time is not local receive time.
- All conclusions remain `SOURCE_TIME_ASSOCIATION`.

### 7.2 Three-stage target

For each decision-eligible row:

1. `S_H`: no next salient event, suspension, game end, or continuity censor before H.
2. `O_H | S_H`: H has an actual home-outcome trade no more than three seconds stale.
3. Conditional on `S_H & O_H`, model price direction and magnitude.

```text
delta_L_H = actual_home_price_H - actual_home_price_L
tick = venue tick from the applicable rule snapshot

UP       delta_L_H >= tick
DOWN     delta_L_H <= -tick
NO_MOVE  abs(delta_L_H) < tick
```

Matched-control P95 noise is not the direction threshold. It may only create an
auxiliary `ABNORMAL_MOVE` label.

The conditional distribution is:

```text
P(NO_MOVE)
+ P(UP)   × F(abs(delta) | UP)
+ P(DOWN) × F(abs(delta) | DOWN)
```

No trade remains missing. Do not forward-fill or convert it to `NO_MOVE`.

### 7.3 Locked feature blocks

Feature blocks enter in this fixed order:

```text
B0  outer-training empirical/hazard baseline
B1  market time/state/activity
B2  B1 + football game state
B3  B2 + atomic event facts
B4  B3 + Stage A reference/reference gap
```

Decision features may include:

- venue, L, H, pre-event actual home price;
- landmark price and staleness;
- prior 30/60-second actual trade count and size;
- exact game time, home margin, possession, down/distance, field position;
- atomic action/outcome tags, yards, return yards, lead/possession change;
- supported Stage A probability and remaining reference gap.

Exclude endpoint price/activity, future event/revision, final result, unsupported
injury/substitution, and all unavailable historical L2 fields.

## 8. Minimal Model Suite

1. **B0 empirical baseline**
   - Outer-training-only constant/stratified hazard, direction rates, and median
     conditional magnitude.

2. **Regularized logistic**
   - Discrete-time clean-window survival.
   - Conditional fresh-trade observation.
   - Multinomial `DOWN/NO_MOVE/UP`.

3. **Shallow XGBoost challenger**
   - Same survival, observation, and direction heads.
   - Captures nonlinear state/event interactions.

4. **Direction-conditional quantile XGBoost**
   - Separate UP and DOWN magnitude models.
   - Primary quantiles: q10, q25, q50, q75, q90.
   - q05/q95 only when the frozen support gate passes.

Do not add GAM, EBM, Cox, neural sequence models, or an unconditional signed
quantile model as a promotion candidate in this phase.

## 9. Chronological OOF, Calibration, and Weighting

Use complete-game expanding folds:

```text
weeks 1–2  -> validate 3–4
weeks 1–4  -> validate 5–6
weeks 1–6  -> validate 7–8
weeks 1–8  -> validate 9–10
weeks 1–10 -> validate 11–12
```

- Both venues for one game remain in the same fold.
- All preprocessing is trained inside the outer training fold.
- Calibration uses only prequential predictions from training games.
- Direction uses multinomial temperature scaling when supported.
- Survival/observation uses training-only logistic/Platt recalibration.
- Unsupported calibration publishes `RAW_UNCALIBRATED`; it never borrows future weeks.
- Final holdout calibrators are frozen from development OOF only.

Weights:

```text
each game has equal total weight
-> episodes divide that game weight equally
-> valid L/H rows divide episode weight equally
-> multiple tags do not increase weight
```

## 10. Metrics and Promotion

### 10.1 Submodel metrics

- Survival/observation: discrete-time NLL, Brier, calibration, coverage/attrition.
- Direction: multiclass log loss primary, multiclass/classwise Brier co-gate.
- Magnitude: pinball and approximate CRPS primary; interval coverage/width secondary.
- All metrics first aggregate per game, then use game-cluster intervals.

### 10.2 Dual confirmatory gate

A model is promotable only if both gates pass:

1. A pre-locked integrated multi-horizon, equal-game loss improves over B0 and
   its paired game-cluster confidence interval excludes no improvement.
2. The clean `L=3 -> H=30` anchor does not reverse sign and is not worse than B0.

Integrated weights, normalization, missingness handling, and per-game aggregation
must be frozen before any OOF run. The dual gate is intersection-union: failure
of either gate rejects promotion.

If source clock quality cannot support a three-second anchor, all model/factor
claims are descriptive and cannot be promoted.

### 10.3 Named factor claims

- Promotion headline is only the clean `L=3 -> H=30` estimate.
- Full L/H paths are secondary trajectories; do not select the best second.
- Apply BH inside the frozen factor family.
- Report distinct games, unique episodes, UP/DOWN counts, maximum game
  contribution, and leave-one-game-out stability.
- A subtype with inadequate support is `INSUFFICIENT_SUPPORT`; this does not
  invalidate its parent event family.

Matched controls are non-promotional diagnostics only. They require fold-local
construction, SMD <= 0.10, overlap, effective sample size, and concentration
reporting. They cannot define labels, models, or reference truth.

## 11. Kalshi Validation Matrix

Current development evidence includes both venues. Treat validation levels
precisely:

1. Polymarket venue-specific OOF.
2. Kalshi venue-specific OOF.
3. Exact same `game/episode/L/H` paired comparison, clustered by game.
4. Development-only temporal venue transport with a frozen source-venue model and
   no target-venue recalibration.

None of these is an independent game holdout.

Before final validation publish:

```text
153 development IDs, weeks, kickoff times, and batch hashes
81 candidate holdout IDs, weeks, kickoff times, and batch hashes
set intersection
overlap with every old review/holdout/reaction cohort
prior exposure/read audit
holdout reaction access counter = 0
```

After `ShortlistLockV1`, run the eligible pristine holdout once:

- Kalshi-development model -> Kalshi holdout;
- Polymarket-development model -> Polymarket holdout;
- frozen Polymarket-development model -> Kalshi holdout.

If a game was previously used to select a factor, horizon, threshold, or claim,
it cannot be called pristine. After the first holdout read, any model or rule
change must be validated on a future cohort, never by retrying these games.

## 12. Implementation Tasks

### Task 1 — Freeze contracts and publish exact-153 facts

**Files**

- Modify `src/prediction_market/sports/nfl_x16_fact_extraction.py`
- Create `src/prediction_market/sports/nfl_x16_exact153.py`
- Create/modify the focused tests under `tests/sports/`

**Work**

- Implement `EpisodeFactV3`, eligibility fields, orientations, and sequence IDs.
- Replace the old TD/try and adjudication semantics directly; no compatibility wrapper.
- Build a deterministic `game/play -> atomic episode` mapping.
- Publish reconciliation and orphan/duplicate audit.
- Stream one game at a time with default `workers=1`.
- Publish per-game content-addressed objects and a verified batch index.
- Re-run a bounded fixture and compare semantic hashes.

### Task 2 — Publish Stage A reference observations

**Files**

- Modify `src/prediction_market/sports/nfl_reference_value.py`
- Create `src/prediction_market/research/nfl_stage_a_reference.py`
- Create/modify focused tests under `tests/research/`

**Work**

- Verify the model asset SHA before inference.
- Publish every feature's `known_at` and provenance.
- Fail closed on future-derived `receive_2h_ko`.
- Enforce pure pre/post transition and composite exclusions.
- Publish supported and unsupported rows.
- Publish equal-game calibration metrics.

### Task 3 — Replace the V3 reaction panel

**Files**

- Replace `src/prediction_market/research/nfl_x15_landmarks.py`
- Update its focused tests

**Work**

- Join facts, Stage A observations, and actual home-outcome trades.
- Preserve H-missing rows for availability.
- Emit `S_H`, `O_H`, direction, and conditional magnitude targets.
- Apply interval, ambiguity, contamination, and staleness rules.
- Prove endpoint data is absent from decision features.
- Publish attrition by venue/L/H/reason.

### Task 4 — Implement OOF models and calibration

**Files**

- Replace `src/prediction_market/research/nfl_x15_models.py`
- Create `src/prediction_market/research/nfl_x15_calibration.py`
- Create `src/prediction_market/research/nfl_x15_distribution.py`
- Update focused tests

**Work**

- Implement B0, logistic, shallow XGBoost, and conditional quantile XGBoost.
- Implement game-grouped chronological OOF and training-only calibration.
- Publish raw/calibrated probabilities and conditional quantiles.
- Publish explicit support failures instead of constant-label models.

### Task 5 — Model selection and Kalshi validation

**Files**

- Create `src/prediction_market/research/nfl_x15_model_selection.py`
- Create `src/prediction_market/research/nfl_x16_kalshi_validation.py`
- Update focused tests

**Work**

- Run locked B0->B4 ablations.
- Apply equal-game integrated and L3/H30 dual gates.
- Publish Poly/Kalshi venue OOF, exact pairs, and transport diagnostics.
- Publish factor-family BH, LOO, and support audit.
- Prove the holdout access path remains unopened.

### Task 6 — Workbench, expert review, and shortlist

**Files**

- Modify `notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb`
- Modify `src/prediction_market/workbench/factor_lab_master.py`
- Create/replace the V3 offline report builder
- Create `src/prediction_market/research/nfl_x15_shortlist.py`

**Work**

- Show 153-game fact coverage and reconciliation.
- Show Stage A support/calibration and composite exclusions.
- Show Stage B availability, direction, distribution, and full time paths.
- Show Poly/Kalshi paired and transport diagnostics.
- Show exact positive, negative, reversal, disagreement, and excluded cases.
- Materialize expert decisions and `ShortlistLockV1`.
- Render one offline content-addressed Chinese/English report.

### Task 7 — One-time final holdout

**Files**

- Create/replace `src/prediction_market/research/nfl_x15_holdout.py`
- Update focused tests

**Work**

- Publish cohort and prior-exposure audit without reading reaction data.
- Verify lock hash and access counter zero.
- Read the eligible pristine reaction cohort once.
- Run the three frozen validation paths.
- Publish replicated, not-replicated, and insufficient-support outcomes.
- Record the irreversible access event and forbid reuse for tuning.

## 13. Focused Verification

Do not run unrelated legacy test suites. Required checks are:

1. DAL–DET bounded raw-to-`EpisodeFactV3` fixture.
2. TD/PAT/2PT, pick-six, retry, review/reversal, no-play, safety, and OT fixtures.
3. Actor/beneficiary/home orientation and actual-home-leg invariants.
4. Exact-153 zero orphan/duplicate/silent-loss mapping audit.
5. Stage A feature-known-at and pure-transition audit.
6. H-missing preservation and no endpoint leakage.
7. Same-second ambiguity and next-event censoring.
8. One-tick direction boundary.
9. Game-grouped OOF and training-only calibration.
10. Equal-game weights and no multi-tag sample explosion.
11. Kalshi exact-pair referential integrity.
12. Holdout access counter remains zero before lock.
13. Master Notebook/report loads only published content-addressed artifacts.

## 14. Completion

The phase is complete only when:

- all 153 games have verified `EpisodeFactV3` bundles;
- Stage A publishes reproducible supported/unsupported reference rows;
- Stage B publishes OOF availability, direction, and distribution for both venues;
- the dual promotion gate and named-factor audits are published;
- Kalshi venue-specific, paired, and transport results are complete;
- the expert shortlist is locked;
- the eligible pristine holdout is read once and reported;
- every row and conclusion traces back to source, model, fold, and lock hashes;
- historical results remain explicitly non-executable source-time research.
