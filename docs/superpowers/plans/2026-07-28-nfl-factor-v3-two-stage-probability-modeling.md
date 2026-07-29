# NFL Factor V3 Two-Stage Probability Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-extract exhaustive, auditable NFL facts for the frozen 153-game development sample and use them in a two-stage model that separates football reference value from the subsequent Polymarket/Kalshi winner-market probability path.

**Architecture:** Stage A produces a version-pinned football win-probability reference before and after each finalized information episode. Stage B consumes the episode facts, reference change, and decision-time market state to predict the conditional distribution of future actual-trade price changes at multiple landmarks and horizons. All model selection uses game-grouped chronological out-of-fold predictions; the 81-game final holdout remains unread until factor, feature, target, and model locks are frozen.

**Tech Stack:** Python 3.12, pandas, PyArrow, DuckDB, scikit-learn 1.9, XGBoost 3.3, frozen nflverse/fastrmodels assets, Jupyter, immutable content-addressed artifacts.

---

## 1. Current State and Non-Negotiable Boundaries

The implementation must reuse, not rebuild, the following published inputs:

- 153 verified X-13 development game bundles.
- 4,773,299 historical actual-trade observations.
- 610,932 historical reaction paths.
- Frozen nflverse play-by-play and participation objects.
- Frozen fastrmodels no-spread XGBoost asset.
- Existing X-15 landmark, walk-forward, policy, and publication machinery.
- Existing S3 hydrate and SHA-256 verification path.

The following old outputs remain immutable evidence but are not training authority for V3:

- X-13 factor registry V2 and its 101-factor distributions.
- X-15 coarse primary-event results.
- Any old report that grouped heterogeneous events such as ordinary `PASS`.

The implementation must not:

- read the 81-game final holdout before `ShortlistLockV1`;
- use random row splits;
- count multiple plays or venues from one game as independent games;
- forward-fill missing historical trades;
- use endpoint information in a decision-time feature;
- call historical actual trades executable bid/ask prices;
- infer injury, substitution, route, coverage, or pressure facts not present in an approved source;
- calculate OFI, depth, queue, or fill metrics from trades-only history.

## 2. First-Principles Research Decomposition

The target is not one probability. It is two linked but distinct questions:

### Stage A — Football reference value

Given finalized game state and episode facts:

```text
p_ref_before = P(home wins | state before episode)
p_ref_after  = P(home wins | state after finalized episode)
reference_delta = p_ref_after - p_ref_before
```

This answers how much the football state changed according to a reproducible open model. It is a diagnostic reference, not ground truth.

### Stage B — Market reaction

At a decision landmark `L` after the episode, predict the actual-trade price change through endpoint `H`:

```text
target_delta(L,H) = focal_trade_price(H) - focal_trade_price(L)
remaining_reference_gap(L) = p_ref_after_focal - focal_trade_price(L)
```

This answers:

- whether the next observed move is up, flat, or down;
- the expected range, not only the mean;
- how much of the reference change is completed by each horizon;
- whether the path continues, stalls, overshoots, or reverses.

The models must never collapse these two questions into one opaque “alpha model.”

## 3. What Mature Practice Contributes

The implementation adopts the following established ideas:

1. **State-value modeling.** Football win probability is conditioned on score, clock, possession, down, distance, field position, timeouts, and pregame strength. The official nflfastR interface exposes `wp` and `vegas_wp`, and published sports-analytics reviews treat state value and win probability as the standard starting point.
   - https://nflfastr.com/reference/calculate_win_probability.html
   - https://arxiv.org/abs/2301.04001

2. **Event surprise, not event name alone.** In-play market research commonly distinguishes an event from how surprising or valuable it was in the pre-event state. An interception in Q1 and one in a Q4 close game therefore share an event subtype but not a predicted impact.
   - https://www.sciencedirect.com/science/article/pii/S0167268114000481

3. **Time-conditional calibration.** A market price cannot be treated as a universally calibrated probability without conditioning on time and product. Recent Kalshi sports evidence reports materially different calibration close to expiry, so game time and pre-event price remain continuous model inputs.
   - https://arxiv.org/abs/2607.14430

4. **Probability calibration on disjoint data.** A classifier’s raw `predict_proba` is not automatically calibrated. Calibration must use predictions from data disjoint from model fitting.
   - https://scikit-learn.org/stable/modules/calibration.html

5. **Distributional prediction.** Signed price changes are not Beta distributed. Quantile regression produces conditional ranges without imposing a symmetric Gaussian shape. XGBoost supports `reg:quantileerror` with histogram trees; quantile crossing must be audited.
   - https://xgboost.readthedocs.io/en/release_3.2.0/python/examples/quantile_regression.html

6. **Chronological validation.** Sports states and venue liquidity change through a season. All tuning, calibration, and evaluation must preserve complete games and time order.

The implementation explicitly does not adopt L2-only methods such as OFI, microprice, DeepLOB, or queue models for the 2025 historical study. Those belong to X-14 prospective L2 data.

## 4. Canonical Fact and Factor Design

### 4.1 Separate facts from research predicates

`CanonicalEpisodeFactV3` is the model input. `FactorDefinitionV3` is a human-readable predicate over those facts.

```python
@dataclass(frozen=True)
class CanonicalEpisodeFactV3:
    game_id: str
    episode_id: str
    event_anchor_utc: datetime
    primary_action: str
    outcome_tags: tuple[str, ...]
    beneficiary_team: str | None
    quarter: int
    game_seconds_remaining: int
    score_margin_beneficiary: int
    down: int | None
    distance: int | None
    yardline_100: float | None
    yards_gained: float | None
    air_yards: float | None
    yards_after_catch: float | None
    return_yards: float | None
    kick_distance: float | None
    possession_changed: bool
    score_delta_beneficiary: int
    confirmed_injury: bool
    source_hashes: tuple[str, ...]
```

`FactorDefinitionV3` remains deterministic:

```python
@dataclass(frozen=True)
class FactorDefinitionV3:
    factor_id: str
    version: str
    family: str
    meaning_en: str
    meaning_zh: str
    predicate: Mapping[str, object]
    exclusions: tuple[str, ...]
    status: Literal["ACTIVE", "DATA_GAP", "REJECT"]
```

The model receives continuous state. It must not create separate sparse model columns for every manually named combination.

### 4.2 Exhaustive episode facts

The 153-game extraction must cover:

- action: pass, run, scramble, sack, punt, kickoff, field goal, try, kneel, spike;
- pass: complete/incomplete, depth, location, air yards, YAC, passer/receiver;
- run: designed run/scramble, location, gap, rusher, yards;
- turnover: interception, lost fumble, turnover on downs, muff, recovery team, return yards, return touchdown;
- score: passing/rushing/defensive/return TD, FG, PAT, 2PT, defensive 2PT, safety;
- special teams: punt/kickoff/FG outcome, touchback, fair catch, inside-20, onside, blocked kick, return;
- administration: accepted/declined/no-play penalty, review, reversal, timeout, deleted/admin rows;
- state: exact remaining seconds, score, margin, possession, down/distance, field coordinate, red zone, timeouts;
- player evidence: stable IDs, explicit roles, approved participation, explicit in-play injury/return evidence.

Pregame injury timelines and exact substitution timestamps remain `DATA_GAP`.

### 4.3 Episode finalization

The feature builder must use the first state before and the final state after the full information episode:

- TD absorbs PAT/2PT, directly related penalty/retry, and review.
- Pick-six retains both turnover and score tags.
- No-play and reversed outcomes do not create live outcome facts.
- Review rows attach to the affected live episode.
- The next salient episode defines contamination, not a future feature.

## 5. Stage A Model Design

### 5.1 Canonical model

The canonical Stage A model is the already-pinned official fastrmodels no-spread XGBoost booster:

```text
model asset: wp_model.ubj
runtime: xgboost==3.3.0
input orientation: possession probability converted to home/focal
formal support: regulation only
OT: MODEL_SUPPORT_UNPROVEN
```

No new football model is trained in V3. This prevents the 153 market-covered games from being misused as a small football-model training set.

### 5.2 Comparators

Publish three Stage A references without blending them:

1. `no_spread_open_model`: canonical reproducible fastrmodels inference.
2. `upstream_vegas_wp_diagnostic`: nflverse-provided value when its lineage is present.
3. `pre_event_market_probability`: actual venue trade, used only as a market baseline.

The system must not average these into a consensus in V3.

### 5.3 Stage A evaluation

For each supported regulation state, compare predicted home probability with final game outcome, but aggregate uncertainty by game.

Report:

- Brier score;
- log loss;
- calibration slope/intercept;
- reliability curve by 10-point probability bins;
- performance by quarter and remaining-time band;
- disagreement between no-spread reference, upstream Vegas diagnostic, and venue prices.

Do not treat the many states within one game as independent observations. Use equal-game weighting and game-cluster bootstrap intervals.

## 6. Stage B Dataset and Targets

### 6.1 Grain

The training panel grain is:

```text
game
× finalized episode
× venue
× beneficiary winner market
× decision landmark
× future endpoint
```

Decision landmarks:

```text
1, 2, 3, 5, 10 seconds
```

Endpoints:

```text
5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 seconds
```

Only endpoints later than the decision landmark are emitted.

### 6.2 Observed marks

Each landmark and endpoint uses an actual focal-outcome trade no more than three seconds old. A missing mark remains missing.

```python
decision_eligible = (
    landmark_trade_observed
    and landmark_orientation_valid
    and not contaminated_before_landmark
    and not censored_before_landmark
)

target_eligible = (
    decision_eligible
    and endpoint_trade_observed
    and endpoint_orientation_valid
    and not contaminated_before_endpoint
    and not censored_before_endpoint
)
```

`decision_eligible` may not depend on endpoint observability.

### 6.3 Targets

Continuous target:

```text
target_delta = endpoint_price - landmark_price
```

Direction target:

```text
DOWN     if target_delta < -noise_threshold
NO_MOVE  if abs(target_delta) <= noise_threshold
UP       if target_delta > noise_threshold
```

`noise_threshold` is trained separately by venue and outer fold:

```text
max(1 percentage point, matched-control P95 absolute change)
```

Auxiliary targets:

- `abs_move_ge_1pp`, `abs_move_ge_2pp`, `abs_move_ge_5pp`;
- continuation versus reversal from the 10-second direction to the endpoint;
- completion fraction only when `abs(reference_delta) >= 1pp`.

## 7. Stage B Feature Lock

### 7.1 Categorical features

- primary action;
- atomic outcome tags;
- score method;
- turnover subtype;
- possession beneficiary;
- venue;
- landmark and endpoint;
- quarter only as an interpretable companion to continuous time.

### 7.2 Numeric/boolean features

- exact game seconds remaining;
- score margin in beneficiary orientation;
- down, distance, yardline, red-zone and goal-to-go;
- yards gained, air yards, YAC, return yards, kick distance;
- possession change, score change, lead-change type;
- timeouts remaining;
- reference probability before/after and `reference_delta`;
- decision-time actual trade price;
- `remaining_reference_gap`;
- decision-time staleness;
- prior 30/60-second actual trade count and size;
- both-venues-active status.

### 7.3 Excluded features

- endpoint price or endpoint activity;
- post-landmark future trade counts;
- final game result;
- future revisions;
- inferred injury or substitution;
- player/team season aggregates in V3;
- formation/personnel strategy in V3;
- historical L2 fields that do not exist.

## 8. Stage B Model Suite

All models train separately by venue. Polymarket and Kalshi may only be pooled in an explicit later experiment with venue interactions.

### B0 — Intercept and matched empirical baselines

Publish:

- unconditional venue/landmark/endpoint base rates;
- nearest matched historical episode distribution;
- factor-level empirical-Bayes shrinkage toward its parent event family.

The matched state uses pre-event probability, remaining time, score margin, field position, staleness, and activity. This is the transparent baseline against which ML must improve.

### B1 — Calibrated multinomial logistic direction model

Predict:

```text
P(DOWN), P(NO_MOVE), P(UP)
```

Use regularized multinomial logistic regression with one-hot categorical and standardized numeric inputs.

For each outer fold:

1. fit the base classifier on the earlier portion of its training weeks;
2. calibrate on the latest training week, never on outer validation;
3. use sigmoid calibration as the primary method;
4. publish uncalibrated and calibrated probabilities for audit.

The current X-15 label `CALIBRATED_CLASS_PROBABILITY` must be replaced unless an actual disjoint calibrator was fitted.

### B2 — XGBoost quantile distribution model

Train `reg:quantileerror` with histogram trees for:

```text
q05, q10, q25, q50, q75, q90, q95
```

Use:

```python
quantile_alpha = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
tree_method = "hist"
```

Audit and monotonically project crossed quantiles before publication. Preserve both raw and projected outputs.

### B3 — XGBoost mean-regression comparator

Retain the existing shallow `reg:squarederror` model as a point-estimate comparator. It cannot be presented as a probability distribution.

### B4 — Threshold probability models

Fit calibrated binary logistic models for:

- `P(abs(delta) >= 1pp)`;
- `P(abs(delta) >= 2pp)`;
- `P(abs(delta) >= 5pp)`;
- `P(reversal by H)`.

If an outer training fold lacks both classes, that target/fold is `INSUFFICIENT_SUPPORT`; it is not silently collapsed to a constant prediction.

## 9. Chronological Validation

Use the existing five outer expanding folds:

```text
weeks 1–2  → validate 3–4
weeks 1–4  → validate 5–6
weeks 1–6  → validate 7–8
weeks 1–8  → validate 9–10
weeks 1–10 → validate 11–12
```

Rules:

- complete games stay in one fold;
- the two venues for one game stay in the same fold;
- preprocessing is fitted inside each training fold;
- calibration uses only training weeks;
- noise thresholds use only training matched controls;
- tuning uses training/calibration data, never outer validation;
- final reported development metrics use only out-of-fold rows.

The 81-game holdout is not an additional tuning fold. It is read once after all locks.

## 10. Model Evaluation

### 10.1 Direction probabilities

Primary:

- multiclass log loss;
- multiclass Brier score;
- class-wise Brier score;
- reliability diagram;
- expected calibration error;
- calibration slope/intercept.

Secondary:

- macro F1;
- balanced accuracy;
- per-class precision/recall;
- confusion matrix.

Raw accuracy is not a primary metric because `NO_MOVE` is common.

### 10.2 Conditional distribution

- pinball loss at every quantile;
- central 50%, 80%, and 90% interval coverage;
- interval width;
- median absolute error;
- approximate CRPS from the quantile grid;
- quantile-crossing rate before projection.

### 10.3 Research stability

- game-cluster bootstrap confidence intervals;
- per-game metric distribution;
- leave-one-game-out direction stability;
- maximum single-game contribution;
- performance by venue, event family, quarter, continuous time, score band, pre-price band, and reference-delta magnitude.

### 10.4 Required ablations

Run the same folds for:

```text
A: market state only
B: A + game state
C: B + atomic event facts
D: C + reference_delta / remaining_reference_gap
E: D + activity and staleness
```

The contribution of a feature block is the paired per-game out-of-fold metric difference. A block is useful only if its game-cluster 95% interval excludes no improvement on the primary metric.

## 11. Promotion and Holdout Rules

### 11.1 Model promotion

A Stage B model may become the frozen development model only when:

- log loss and Brier improve over B0 using paired game-cluster intervals;
- calibration is not worse than the uncalibrated baseline;
- improvement occurs in at least four of five outer folds;
- the result is not driven by one game;
- required event/state features pass PIT and provenance checks;
- venue-specific support is adequate.

Quantile publication additionally requires:

- central interval coverage within five percentage points of nominal;
- projected quantiles are monotone;
- median pinball loss improves over the empirical median baseline.

### 11.2 Factor shortlist

A named factor is eligible for expert review only when:

- its sports definition is `ACCEPT`;
- it spans at least 30 games per claimed venue;
- OOF direction and magnitude are stable;
- its effect is not a residual heterogeneous bucket;
- factor-family BH correction and leave-one-game-out gates pass.

No cross-venue requirement is imposed on a venue-specific model. A cross-venue claim requires both venues independently to pass and agree in sign.

### 11.3 Final holdout

`ShortlistLockV1` freezes:

- factor registry hash;
- episode finalization rules;
- feature list;
- landmarks/endpoints;
- targets/noise thresholds;
- model hyperparameters;
- calibration method;
- exclusions;
- metric and promotion gates.

The 81-game holdout is then evaluated exactly once. A claim replicates only when its holdout effect is in the same direction, its 95% interval excludes zero/no improvement, and its magnitude is at least 50% of development.

## 12. Public Interfaces and Artifacts

### Task 1: Publish exact-153 V3 facts

**Files:**
- Modify: `src/prediction_market/sports/nfl_x16_fact_extraction.py`
- Create: `src/prediction_market/sports/nfl_x16_exact153.py`
- Test: `tests/sports/test_nfl_x16_exact153.py`

- [ ] Add an episode-level V3 fact schema without changing the frozen raw inputs.
- [ ] Run the single-game extractor over four deterministic game shards.
- [ ] Reconcile every raw row and every finalized episode.
- [ ] Publish one content-addressed game bundle per game and one batch index.
- [ ] Verify two runs produce identical semantic hashes.

Public entry point:

```python
def build_exact153_fact_bundle_v3(
    *,
    batch_index: Path,
    factor_registry: Path,
    output_root: Path,
    workers: int = 4,
) -> Exact153FactPublication:
    ...
```

### Task 2: Build Stage A reference observations

**Files:**
- Modify: `src/prediction_market/sports/nfl_reference_value.py`
- Create: `src/prediction_market/research/nfl_stage_a_reference.py`
- Test: `tests/research/test_nfl_stage_a_reference.py`

- [ ] Load the pinned fastrmodels asset by hash.
- [ ] Calculate pre/post reference probability for supported regulation episodes.
- [ ] Orient all values to the beneficiary and home outcomes.
- [ ] Publish unsupported/excluded reason rows.
- [ ] Publish game-weighted calibration and comparison tables.

### Task 3: Build the V3 landmark training panel

**Files:**
- Replace: `src/prediction_market/research/nfl_x15_landmarks.py`
- Test: `tests/research/test_nfl_x15_landmarks.py`

- [ ] Join episode facts, reference observations, and actual winner trades.
- [ ] Preserve decision eligibility independently from future target availability.
- [ ] Emit all approved landmark/endpoint pairs.
- [ ] Prove no endpoint field enters the feature view.
- [ ] Publish attrition by venue, landmark, endpoint, and exclusion reason.

### Task 4: Implement calibrated direction models

**Files:**
- Replace: `src/prediction_market/research/nfl_x15_models.py`
- Create: `src/prediction_market/research/nfl_x15_calibration.py`
- Test: `tests/research/test_nfl_x15_models.py`
- Test: `tests/research/test_nfl_x15_calibration.py`

- [ ] Implement outer week folds and inner chronological calibration.
- [ ] Publish raw and calibrated class probabilities.
- [ ] Calculate log loss, Brier, ECE, slope/intercept, and macro F1.
- [ ] Reject training folds without all required classes.
- [ ] Remove the false calibrated-probability label from old behavior.

### Task 5: Implement conditional distribution models

**Files:**
- Create: `src/prediction_market/research/nfl_x15_distribution.py`
- Test: `tests/research/test_nfl_x15_distribution.py`

- [ ] Train the seven-quantile XGBoost model.
- [ ] Preserve raw quantiles and publish monotone projected quantiles.
- [ ] Implement threshold-probability classifiers.
- [ ] Calculate pinball, coverage, width, median MAE, and approximate CRPS.
- [ ] Publish `INSUFFICIENT_SUPPORT` instead of constant-label models.

### Task 6: Run ablations and model selection

**Files:**
- Create: `src/prediction_market/research/nfl_x15_model_selection.py`
- Test: `tests/research/test_nfl_x15_model_selection.py`

- [ ] Run the A–E feature-block ablation matrix.
- [ ] Compare per-game OOF losses with cluster bootstrap intervals.
- [ ] Apply the fixed promotion rules.
- [ ] Publish selected and rejected models with exact reasons.
- [ ] Prove no holdout path is opened.

### Task 7: Publish the single research workbench

**Files:**
- Modify: `notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb`
- Modify: `src/prediction_market/workbench/factor_lab_master.py`
- Create: `src/prediction_market/reports/nfl_x16_probability_workbench.py`
- Test: `tests/workbench/test_nfl_x16_probability_workbench.py`

- [ ] Add V3 fact coverage and V2→V3 comparison.
- [ ] Add Stage A reference calibration.
- [ ] Add Stage B direction probabilities and quantile bands.
- [ ] Add landmark/endpoint selection and state breakdowns.
- [ ] Add per-game cases, exclusions, model cards, and lineage.
- [ ] Render one offline content-addressed HTML without external assets.

### Task 8: Freeze shortlist and run the holdout once

**Files:**
- Create: `src/prediction_market/research/nfl_x15_shortlist.py`
- Create: `src/prediction_market/research/nfl_x15_holdout.py`
- Test: `tests/research/test_nfl_x15_holdout.py`

- [ ] Materialize expert review decisions.
- [ ] Generate `ShortlistLockV1` and its semantic hash.
- [ ] Verify the holdout access counter is zero before the lock.
- [ ] Evaluate the sealed holdout once with the frozen model.
- [ ] Publish replicated, not-replicated, and insufficient-support outcomes.

## 13. Acceptance Criteria

The implementation is complete when:

1. All 153 games have V3 fact and episode bundles with zero silent row loss.
2. The V3 factor registry is frozen after coverage and expert review.
3. Stage A publishes reproducible pre/post reference probabilities with explicit unsupported rows.
4. Stage B publishes OOF direction probabilities and conditional quantiles for both venues separately.
5. Every prediction is traceable to sports, market, factor, model, and fold hashes.
6. The workbench can explain one prediction from raw event through final distribution.
7. Calibration, distribution coverage, ablation, and per-game stability are visible.
8. Old V2/X-15 artifacts remain unchanged.
9. Holdout access remains zero until the shortlist lock, then occurs once.
10. All conclusions retain the boundary: historical actual-trade probability-path research, not executable alpha.
