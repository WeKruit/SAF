# X-16 NFL Source-Anchor Reaction Trajectory

## Objective

Implement a new canonical experiment:

```text
X-16 NFL source-anchor-relative probability trajectory
```

X-16 must not modify or reinterpret X-15. Existing X-15 source contracts,
`mark_L/mark_H`, `NO_MODEL_ADVANCE`, manifests, and reports remain immutable.

The study has two stages:

1. Stage A uses the frozen nflfastR/fastrmodels no-spread XGBoost to produce
   source-provenance-bound football reference probabilities.
2. Stage B estimates the probability that an actual trade is observed in each
   registered three-second endpoint window and, conditional on an actual
   trade, the distribution of the home-contract price.

Kalshi is the primary development, selection, and holdout venue. Polymarket is
a frozen replication venue. Venues are never pooled.

Claim boundary:

```text
PRELIMINARY_SOURCE_TIME_TRADES_ONLY
NON_CAUSAL
NON_EXECUTABLE
NON_ALPHA
```

## Frozen Sources

Reuse existing immutable objects only. Do not fetch upstream data.

| Authority | Frozen evidence |
|---|---|
| Development cohort | 153 games |
| Sealed holdout | 81 games; reaction reads must remain zero before selection lock |
| PIA V2 | 25,250 anchors; 25,070 primary-supported |
| Development market observations | 4,773,299 actual observations |
| PIA manifest | `sha256:fac9db…` |
| Development market batch | `sha256:b21640…` |
| Capture batch | `sha256:ce8e509…` |
| Cohort authority | `sha256:545f7e…` / object `sha256:226b796…` |
| Event prestate context | `sha256:61afab…` |
| Native-rule audit | `sha256:83394c…`; strict cross-venue equivalence is 0/153 |

If an object is absent locally, hydrate only by its manifest hash from the
existing S3 store. Do not access PMXT upstream and do not edit old manifests.

## Canonical Time and Price Contract

Each formal row uses `ProvisionalInformationAnchorV2`:

```text
[S,T] = [source_interval_start, source_interval_end]
H = {3,5,10,15,20,25,30,35,40,45,50,55,60,75,90}

p0  = actual home-outcome trade VWAP in [S-3s,S)
p_h = actual home-outcome trade VWAP in (T+h-3s,T+h]
```

Use every valid trade in the full window and exact Decimal arithmetic. Require
price in `[0,1]`, size `>0`, exact native home contract identity, complete
pagination proof, and unique native trade identity. Never use a candle,
midpoint, inferred complement, away contract, last-trade fallback, or
forward-fill.

Output:

```text
delta_probability_0_h = p_h - p0
delta_pp_0_h          = 100 * (p_h - p0)
delta_logit_0_h       = logit(p_h) - logit(p0)
```

Prices equal to zero or one keep raw/pp values and receive a null logit value.
Do not clip them.

For interval-valued trade timestamps, include a trade only when its complete
timestamp interval is inside the price window. Boundary-crossing records are
`TIMESTAMP_MEMBERSHIP_AMBIGUOUS`.

Publish fixed clock-shift sensitivity at:

```text
-10,-5,-3,-2,-1,0,+1,+2,+3,+5,+10 seconds
```

Do not select a favorable shift. Mark conclusions changing direction inside
`±3s` as `CLOCK_SENSITIVE`.

## Target Roles

### Operational

`OPERATIONAL_EX_ANTE_TARGET` retains later sports information. Later PIA count
and first time are label/audit fields and must be physically absent from model
features. This target is ex-ante at a fixed horizon, but the resulting move
cannot be attributed solely to the initiating event.

### Isolated

`RETROSPECTIVE_ISOLATED_DIAGNOSTIC` is censored when:

```text
next_pia.source_interval_start <= T+h
```

Isolation uses future absence of information and can never be a deployable
feature, model, or holdout-promotion target.

### Continuation

Use:

```text
L = {3,5,10}
H-L >= 3
```

There are exactly 38 `L/H` pairs. Continuation is isolated and descriptive
only.

## Required Missingness States

Keep these states separate:

```text
OBSERVED_ACTUAL_TRADE_VWAP
NO_OBSERVED_ACTUAL_TRADE_IN_VERIFIED_CAPTURE_WINDOW
CAPTURE_COVERAGE_INVALID
NO_BASELINE_TRADE
INITIAL_EVENT_INTERVAL_AMBIGUOUS
PRIMARY_SUPPORT_INELIGIBLE
OUTSIDE_VERIFIED_GAME_OR_MARKET_LIFECYCLE
TIMESTAMP_MEMBERSHIP_AMBIGUOUS
```

Always publish:

```text
historical_market_open_status = UNKNOWN
historical_suspension_status = UNKNOWN_NO_AUTHORITATIVE_CHANNEL
historical_tradeability_status = UNKNOWN
```

An empty endpoint window is not a zero move and does not prove suspension,
closure, illiquidity, or continuous availability.

## Feature and Reference Layers

Publish two separate sports tables:

1. Human-review facts contain full replay and finalized reconciliation.
2. `NFLX16InformationFeatureSnapshotV1` contains only fields with
   `known_at <= T`.

Each formal feature includes its source row, source hash, known-at time,
provisional/final status, model allow flag, and exclusion reason.

Allowed features include continuous game time, quarter, score margin,
possession, down/distance, yardline, red zone, timeouts, provisional event
semantics and magnitude, provisional beneficiary, `p0`, and pre-anchor actual
trade activity.

Player/team fixed effects, player IDs, formation learning, rolling strength,
news, inferred injury severity, tracking, routes, coverage, and every post-T
field are excluded from the formal model.

Stage A publishes `NFLX16ReferenceObservationV1` with:

```text
reference_before_anchor
reference_at_anchor
reference_delta_home
reference_delta_beneficiary
model version and bytes SHA
field-level known_at and source hashes
support status
```

`reference_at_anchor` may use only provisional state proven known by T. Do not
fill it from the next play or final outcome. OT is
`MODEL_SUPPORT_UNPROVEN`.

## Stage B Models

The first head estimates:

```text
P(actual trade observed in the horizon window |
  p0 observed,
  verified capture)
```

It is an observed-trade propensity, not an availability or liquidity model.

Conditional on an observed endpoint trade, output:

```text
q25/q50/q75(p_h)
q10/q90(p_h) only when support passes
q25/q50/q75(delta_pp)
P(delta > 0), P(delta = 0), P(delta < 0)
P(abs(delta) >= 0.5/1/2/5pp)
P(material move up | abs(delta) >= 1pp)
```

Use two formal feature projections:

- `B0_market`: horizon, `logit(p0)`, pre-anchor trade counts, sizes, last-trade
  age, 30/60-second activity, timestamp resolution, and game time.
- `M1_full`: B0 plus decision-time state, provisional event semantics and
  magnitude, provisional beneficiary, and PIT-proven nflfastR reference
  fields.

All horizons share one pooled-horizon model. Do not fit factor-by-horizon
models or third-order interactions.

Model families:

- observed-trade propensity: elastic-net logistic, `l1_ratio=.5`,
  `C={.01,.1,1,10}`;
- direction/materiality: secondary elastic-net binary/multinomial models;
- conditional distribution: L1-regularized quantile regression at
  `.10/.25/.50/.75/.90`, `alpha={1e-4,1e-3,1e-2,1e-1}`.

Use blocked inner CV and the one-SE rule. Save raw quantiles and separately
publish monotone rearrangement. Convert logit-delta quantiles back through
`expit(logit(p0)+q)` before calculating pp.

## Validation and Holdout

Outer folds:

| Fold | Train | Validate |
|---|---|---|
| F1 | Week 1-4 | Week 5-6 |
| F2 | Week 1-6 | Week 7-8 |
| F3 | Week 1-8 | Week 9-10 |
| F4 | Week 1-10 | Week 11-12 |

Keep every game, both venues, and all horizons in the same fold. Fit all
preprocessing and calibration inside outer-train. Weight:

```text
horizon within anchor -> anchor within game -> equal-weight game
```

Primary loss is the average q25/q50/q75 pinball loss (`Q3`). Also publish 50%
interval score/coverage, observed-trade log loss/Brier, conditional direction
log loss, and ordered RPS.

`M1_full` advances only if all development gates pass:

1. Kalshi operational paired Q3 improvement is at least 2%.
2. One-sided 95% paired game-bootstrap lower bound is positive.
3. Observed-trade log-loss skill lower bound is at least -1%.
4. q25/q50/q75 hit-rate errors are each within .05.
5. 50% interval coverage error is within .05 and its CI contains .50.
6. At least 3/4 folds and 10/15 horizons have non-negative Q3 skill.
7. No supported horizon is worse than B0 by more than 10%.
8. Leave-one-game-out total skill always remains positive.
9. All source, PIT, window, orientation, and holdout-read gates hard-pass.

Failure publishes `NO_INCREMENTAL_MODEL_ADVANCE`; Polymarket, direction,
isolated, and continuation cannot rescue it.

Only a passing, hash-locked development model may read the 81-game holdout.
Otherwise keep holdout reaction reads at zero. Holdout is one-time and cannot
be used to revise the model.

## Artifacts and Public API

Create new X-16 modules only; do not add an X-15 adapter:

```python
verify_x16_sources(...)
build_x16_information_feature_snapshot(...)
build_x16_reference_observations(...)
build_x16_reaction_trajectory_game(...)
publish_x16_exact153_trajectory(...)
run_x16_operational_oof(...)
freeze_x16_selection_lock(...)
run_x16_holdout_once(...)
render_x16_report(...)
```

Publish:

```text
NFLX16InformationFeatureSnapshotV1
NFLX16ReferenceObservationV1
NFLReactionEventAuditV1
NFLReactionHorizonTargetV1
NFLReactionContinuationTargetV1
NFLReactionTrajectoryAttritionV1
NFLReactionOOFPredictionV1
NFLReactionModelScoreV1
NFLReactionSelectionLockV1
NFLReactionHoldoutResultV1
```

Output root:

```text
artifacts/market-observation/nfl/x16/reaction-trajectory-v1/
```

Hard reconciliation:

```text
25,250 total anchors
25,070 primary-supported anchors
752,100 horizon candidates
1,905,320 continuation candidates
```

Process one game at a time using sorted timestamp arrays and prefix sums.
Never concatenate all raw market observations. Mirror published objects and
manifests to the existing content-addressed S3 namespace and verify read-back
SHA.

## Research Output

Update the single authoritative notebook:

```text
notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb
```

The notebook and offline content-addressed report must show:

- source/hash/holdout status;
- factor and PIT coverage;
- p0/p_h attrition;
- Kalshi observed-trade and probability-change heatmaps;
- conditional quantile bands with games/anchors/support per cell;
- continuous time/state breakdown;
- operational versus isolated and continuation;
- Stage A reference provenance and diagnostics;
- B0 versus M1 OOF metrics, calibration, bootstrap, and LOO;
- fixed Polymarket replication;
- matched controls, pre-anchor placebos, pseudo-anchors, and clock shifts;
- positive, negative, large, clock-sensitive, and venue-disagreement cases.

No path claim may combine changing survivor cohorts. A progressive/reversal
claim requires a common complete-case risk set through the maximum horizon.

## Focused Verification

Write failing tests first for:

1. exact four endpoint boundaries;
2. full-window VWAP versus old last-timestamp behavior;
3. empty windows remaining null;
4. capture gap versus verified empty;
5. interval-overlap and timestamp ambiguity;
6. operational future fields being physically absent from features;
7. isolated equality boundary censoring;
8. exact 38 continuation pairs;
9. home contract and native trade identity;
10. raw/pp/logit separation and no endpoint clipping;
11. future-event/trade/final-result mutation invariance;
12. game-grouped chronological folds and equal-game weighting;
13. quantile order and calibration;
14. holdout read counter remaining zero before lock;
15. two identical development runs yielding the same batch SHA;
16. report/notebook never invoking raw loaders or model fitters.

Do not run unrelated large test suites.

## Mandatory Report Language

> This study estimates historical home-contract actual-trade VWAP paths
> relative to source-reported NFL information anchors and estimates price
> distributions only when an actual trade occurred in the stated three-second
> window. It does not estimate a latent price when no trade occurred,
> continuous venue availability, causal reaction latency, executable returns,
> or alpha. `p0` is a historical pre-anchor baseline, not a post-information
> executable entry price.
