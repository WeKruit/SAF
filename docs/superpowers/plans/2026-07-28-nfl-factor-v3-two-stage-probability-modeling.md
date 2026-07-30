# NFL Exact-153 Corrected Two-Stage Probability Study

> Status: `CONDITIONAL_GO`
>
> The old event-row PanelV2 and its partial exact-45 run are immutable
> diagnostics only. They cannot select a model, freeze a shortlist, or unlock
> the 81-game holdout.

## 1. Research questions

This phase answers two separate questions.

### Stage A — football reference state

Given only the NFL state known before a provisional information anchor, what is
the frozen no-spread fastrmodels estimate of home win probability?

Stage A is a retrospective, regulation-only diagnostic reference. It is not
ground truth, a live feed, or an executable fair price.

### Stage B — residual market path

At `L` seconds after the information anchor is first known, using the actual trade
available at `L`, what is:

1. the probability that at least one new actual trade appears by `H`; and
2. conditional on a new trade, the probability of `DOWN / NO_MOVE / UP` from
   the `L` mark to the final actual trade in `(L,H]`?

Stage B begins after `L`. It does not estimate the complete pre-event to
post-event shock and does not establish causality, latency, execution, or
tradable alpha.

## 2. Why the previous exact-45 is blocked

The previous run is retained, but not promoted, for four verified reasons:

1. **OT state collision**
   - 10,235 decision rows were OT.
   - OT `game_seconds_remaining=0..600` overlaps Q4.
   - the old D2/D3 inputs had no OT discriminator.
   - Stage A itself is regulation-only.
2. **Stage A prestate was hidden**
   - the old all-or-nothing landmark helper hid `p_before_home` whenever the
     post-state was not yet known;
   - 737,860 old decision rows had a potentially usable prestate reference;
   - `p_after_home`, `reference_delta_home`, and `reference_gap` remain
     prohibited at decision time.
3. **The old “episode” was one source row**
   - `atomic_information_episode_id == event_id`;
   - TD/try and adjudication chains were not actually finalized;
   - old market marks were therefore anchored to row events, not final
     information episodes.
4. **Factor representation was duplicated**
   - D3 contained `primary_action`, `event_tag__*`, and synonymous
     `factor__*`;
   - candidate performance could not be attributed to an individual factor.

The old run stopped after 27 supported cells. These shards remain
content-addressed diagnostics and are never mixed with the corrected run.

## 3. Reused immutable inputs

No sports or market source is downloaded again.

| Input | Authority |
|---|---|
| Exact-153 Facts V4 | 153 games, 26,192 reconciled rows, 83,659 hits, zero silent loss |
| Stage A v1 | frozen fastrmodels no-spread XGBoost bytes, state predictions, 11-field PIT provenance |
| Exact-153 market batch | verified Polymarket and Kalshi actual trades and exact contract orientation |
| Cohort authority | exact game, week, kickoff, batch, and development identity |
| Fold authority | expanding W1–2→W3–4 through W1–10→W11–12 |
| Storage | existing immutable objects, manifests, hashes, and S3 policy |

PanelV2, Stage A event-level reference observations, and old exact-45 outputs
are comparison evidence only.

## 4. Corrected data flow

```text
Facts V4 canonical rows
→ ProvisionalInformationAnchorV2 + FinalizedEpisodeV2 + reconciliation
→ regulation-only information-anchor filter
→ EventPrestateContextV2
→ verified normalized actual trades + contracts
→ RegulationDecisionPanelV4
→ corrected 45-cell chronological OOF
→ Polymarket-only frozen winner
→ Kalshi frozen-winner transport
→ semantic-group ablation + expert review
→ ShortlistLock
→ one-time 81-game holdout
```

No step reads holdout reaction data before `ShortlistLock`.

## 5. Information anchors and finalized episodes

### 5.1 Constituent rules

Every canonical Facts V4 row must map to exactly one:

- finalized episode;
- explicit audit-only administrative record; or
- explicit data-gap exclusion.

No row can silently disappear.

Allowed linkage is structural only:

- stable event/play ID;
- `score_sequence_id`;
- stable adjudication/review linkage;
- source-provided constituent reference.

Description text, row adjacency, and LLM inference cannot create linkage.

### 5.2 Episode rules

- A clean live play may be one episode.
- A TD chain absorbs its structurally linked PAT/2PT attempt, retry,
  penalty, and review; it does not absorb kickoff.
- A game-ending TD without a try requires explicit game-end evidence.
- A review/reversal keeps only the final effective outcome.
- A nullified/no-play episode cannot retain the original TD/turnover label.
- An unresolved adjudication chain is
  `UNRESOLVED_ADJUDICATION_CHAIN / DATA_GAP`.
- `quarter >= 5` is retained for audit and excluded from this experiment.

### 5.3 ProvisionalInformationAnchorV2

Every mapped regulation constituent/information event publishes exactly one
primary anchor row:

```text
information_anchor_id
episode_id
event_id
information_role
source_interval_start
source_interval_end
information_anchor
provisional_* snapshot fields
primary_selection_eligible
censor_boundary_eligible
source hashes
```

A TD and its subsequent PAT/2PT are separate primary anchors linked to the
same finalized parent. Timeout, correction, nullified, unresolved, data-gap,
and unsupported review/reversal anchors remain censor boundaries even if they
cannot provide a formal target.

The primary window is censored from the next chronological provisional
information interval's start, including the next constituent inside the same
parent episode. If source intervals overlap, ordering is ambiguous and the row
is excluded. The censor never waits for the next interval end or parent
episode finalization.

### 5.4 Time anchors

The exact-153 V1 audit triggered ADR 0007. A TD chain's final-known anchor
followed the TD by a median 49.44 seconds, so one anchor cannot represent both
the principal event reaction and the post-try residual.

```text
episode_interval_start
  = first constituent source interval start

episode_known_interval_end
  = last constituent interval end needed to know final semantics

PROVISIONAL_FIRST_SEEN
  = first constituent source interval end

FINALIZED
  = episode_known_interval_end
```

The historical source-time interval is not converted into a fake receive
timestamp.

The formal 45-cell selection consumes only:

```text
source_table = ProvisionalInformationAnchorV2
analysis_role = PRIMARY_SELECTION
```

The final-known anchor is:

```text
source_table = FinalizedEpisodeV2
analysis_role = RESIDUAL_CORRECTION_DIAGNOSTIC
```

It cannot be pooled with, ranked against, or substituted for the primary
anchor.

Primary decision features are a separately hashed constituent snapshot:

```text
provisional_known_at
provisional_primary_action
provisional_outcome_tags
provisional_actor_team
provisional_beneficiary_team
provisional_score_points_observed
provisional_turnover_observed
provisional_yards_gained
provisional_return_yards
provisional_feature_support_status
provisional_feature_support_reason
provisional_snapshot_sha256
```

Panel V4 maps these to the D3 semantic groups and carries:

```text
primary_feature_known_at
primary_feature_support_status
primary_feature_support_reason
primary_feature_snapshot_sha256
```

Formal model rows require `primary_feature_support_status = SUPPORTED` and
`primary_feature_known_at <= information_anchor`, where
`information_anchor == source_interval_end`. Final outcome, PAT/2PT,
review/reversal resolution, final beneficiary, and final score/turnover fields
are forbidden primary inputs. A retrospectively revised source row without an
auditable revision sequence is `UNPROVEN`: it remains an event/censor boundary
but not a model row. The runner may not condition inclusion on whether an
episode later became `FINAL`.

V2 uses three non-interchangeable gates:

```text
primary_selection_eligible
residual_diagnostic_eligible
censor_boundary_eligible
```

No generic `stage_b_eligible` column is retained. Primary eligibility is known
at the provisional anchor, residual eligibility may depend on the parent final
outcome, and every first-seen information anchor remains a censor boundary.

### 5.6 Calibration and hierarchical weighting

For learned models in each outer chronological fold:

```text
base-fit weeks
→ strictly later calibration week
→ outer validation weeks
```

The outer predictions use the same fitted model/preprocessor that produced the
calibration scores. A post-calibration refit is prohibited. Each prediction
and calibrator manifest records the shared producer hashes, and calibration
games cannot overlap validation.

For each target risk set:

```text
w_r = 1 / (E_g * R_ge)
```

where `E_g` is the number of eligible provisional information anchors in game
`g` and `R_ge` is the number of eligible L/H rows for anchor `e`. Direction
recomputes weights only
where `availability_h = 1`. Weighted numeric imputation and scaling use these
same weights; categorical missing values use a fixed token. A duplicate-row
invariance fixture must prove that copying every row of one information anchor changes
neither preprocessing nor predictions.

Evaluation is:

```text
row loss → mean within information anchor → mean within game → mean across games
```

Candidate-minus-B0 is paired on exact source-row identity before the same
aggregation. Confidence intervals resample games, not rows or episodes.

For both analysis roles, a reaction window is censored from the next
provisional information interval's start, never at its interval end or parent
episode's finalized time. A PAT/2PT or review in the same finalized episode
therefore censors the preceding TD. Timeout, unresolved adjudication, data-gap,
correction, and nullified anchors remain boundaries even when they cannot
supply a model target. Overlapping intervals are order-ambiguous and excluded.

### 5.4 Final-outcome corrections

- `review_result=reversed` does not by itself nullify the final structured
  outcome;
- an unlinked reversal remains `DATA_GAP`;
- timeout is a valid Stage-B event and censor boundary even when it is not a
  Stage-A sports-outcome row;
- kickoff-return TD participates in its structured score sequence;
- Stage-B eligibility is independent of Stage-A eligibility.

## 6. Stage A corrected reference

### 6.1 Prestate source

`p_before_home` comes from the Stage A `state_predictions` row that matches the
provisional information anchor's source event by:

```text
game_id
+ event_id / state_event_id
+ raw_play_id / state_raw_play_id
+ information_anchor_id / source event identity
```

The matched state row is itself the event prestate. Its own:

- `reference_status`;
- `state_known_at`;
- `p_home`;
- `state_input_sha256`;
- model hashes; and
- 11 PIT feature-provenance rows

are verified independently.

Availability must not depend on the later post-state or the composite
`reference_observations.reference_status`.

`EventPrestateContextV2` publishes exactly one row for every verified
`ProvisionalInformationAnchorV2` row. It contains only event-prestate fields:

```text
information_anchor_id
event_id
quarter / is_overtime
game_seconds_remaining
home_team / away_team
score_margin_home
possession_is_home
down / distance / yardline_100
goal_to_go / pre_red_zone
home_timeouts_remaining / away_timeouts_remaining
p_before_home / p_before support and missing indicator
prestate_known_at
fact, state-input, model, and feature-provenance hashes
```

Every nonmissing value must be known no later than `information_anchor`.
Post-event scores, next state, finalized semantics, market values, and holdout
data are forbidden. Missing Stage-A support remains explicit and does not
remove the information anchor.

### 6.2 Allowed and forbidden decision features

Allowed:

```text
p_before_home
p_before_support
p_before_missing_indicator
pre_state_known_at
```

Forbidden:

```text
p_after_home
reference_delta_home
bridge_delta_home
reference_gap_at_landmark
any future post-state status
```

### 6.3 Finalized episode reference

`p_after_home/reference_delta_home` may be published for retrospective episode
diagnostics only when a clean poststate is known at the episode anchor. It is
never a Stage B decision feature.

Stage A metrics remain equal-game weighted:

- Brier;
- log loss;
- calibration slope/intercept;
- game-cluster bootstrap CI.

## 7. RegulationDecisionPanelV4

### 7.1 Time grid

```text
L ∈ {1, 2, 3, 5, 10}
H ∈ {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60}
H > L
```

### 7.2 Actual-trade marks

For each provisional information anchor × venue × actual home moneyline
contract, the canonical proposition is:

```text
NFL_FULL_GAME_WINNER / HOME_TEAM_OUTCOME
```

The sports event anchor is regulation-only, but the observed market contract
settles the full game, normally including overtime. The model and report must
not rename that proposition as a regulation winner.

```text
mark_L
  = latest actual-trade timestamp bucket at or before T_selected_anchor + L,
    strictly after the event source interval

availability_H
  = at least one new actual trade in (T_L, T_H]

mark_H
  = last actual-trade timestamp bucket in (T_L, T_H]
```

Same-timestamp trades are size-weighted within one unordered source-time
bucket. Missing trades are never forward-filled.

### 7.3 Censoring

If the next provisional information interval starts, or the verified capture
window/pagination coverage is invalid by `H`, the row is censored and excluded
from Stage B loss. This includes a linked PAT/2PT/review inside the same
finalized episode. Censoring is not relabeled as “no trade” and is not a
selectable prediction head.

Historical trade capture has no authoritative suspension channel and does not
prove continuous open-book state. Therefore this experiment cannot censor on,
or infer availability through, venue suspension/resume or L2 continuity.
Those are X-14 prospective-L2 capabilities. Here `availability_H=0` means only
`NO_NEW_OBSERVED_TRADE` inside a verified captured-trade window; it is not
evidence that the contract was tradable, open, or liquid.

### 7.4 Targets

```text
availability_H ∈ {0,1}

direction_H | availability_H=1
  ∈ {DOWN, NO_MOVE, UP}

NO_MOVE iff abs(mark_H - mark_L) < 0.01
```

The primary joint row loss is:

```text
NLL(availability_H)
+ I(availability_H) × NLL(direction_H)
```

Availability Brier/log loss and direction multiclass Brier/log loss are also
reported separately.

## 8. Orthogonal feature blocks

The corrected matrix remains 45 cells.

| Block | Inputs |
|---|---|
| D0 | `L`, `H` |
| D1 | D0 + `mark_L`, staleness, prior 30/60s actual-trade count and size |
| D2 | D1 + regulation prestate, continuous time, score, possession, down/distance, field position, PIT-safe `p_before_home` and missing indicator |
| D3 | D2 + mutually exclusive provisional-information semantic axes and continuous observed fields |

D3 does not read duplicate `factor__* + event_tag__*` columns.

Every row carries `information_anchor_id`, `episode_id`, `anchor_kind`, and
`analysis_role`. The formal model runner fails closed unless all rows come from
`ProvisionalInformationAnchorV2` and are
`PROVISIONAL_FIRST_SEEN + PRIMARY_SELECTION`.

Semantic axes:

- `action_group`;
- `possession_result_group`;
- `score_result_group`;
- `kick_result_group`;
- `adjudication_group`;
- actor/beneficiary orientation;
- yards, return yards, score margin, seconds remaining, down/distance, and
  field position.

Factor definitions remain available for audit and cases. They are not
duplicated model columns.

## 9. Factor review state

Independent sports review classified the 59 V4 definitions:

- 46 `ACCEPT`;
- 7 `REDEFINE`;
- 5 `DATA_GAP`;
- 1 `REJECT_STANDALONE`.

Required redefinitions:

- return TD → pick-six, fumble-, punt-, kickoff-return TD;
- penalty → accepted/declined/offsetting/no-play + team + resulting state;
- timeout → calling team + remaining timeouts + clock/margin;
- no-play → original outcome + final resulting state;
- injury/return → team + player/position + status;
- late score → score subtype + score sequence de-duplication.

Review/reversal remains a data gap until final adjudication linkage and
known-time are proved. Generic touchback remains an audit roll-up, not an
independent hypothesis.

## 10. Models and folds

Models:

- `b0_empirical_v1 / D0`;
- regularized logistic D0–D3;
- shallow XGBoost D0–D3.

Folds:

| Fold | Train | Validate |
|---|---|---|
| 01 | W1–2 | W3–4 |
| 02 | W1–4 | W5–6 |
| 03 | W1–6 | W7–8 |
| 04 | W1–8 | W9–10 |
| 05 | W1–10 | W11–12 |

For logistic/XGBoost, the last training week is a strictly later calibration
split; the base model/preprocessor is fit on preceding training weeks and is
not refit afterward. B0 uses all outer training weeks and no learned
calibrator.

Weighting:

1. equal L/H rows within provisional information anchor;
2. equal information anchors within game;
3. equal games in aggregate.

Inference uses game-cluster bootstrap. Polymarket and Kalshi rows for the same
information anchor are not independent sports events.

Only one heavy model process runs on the 16-GiB host. Each cell publishes
atomically and can resume after byte, schema, semantic, and provenance
verification.

## 11. Polymarket selection

Polymarket is the only model-selection venue.

Candidate and B0 must use exact paired rows. A candidate advances only if:

- joint-loss improvement 95% game-bootstrap CI lower bound > 0;
- direction log-loss improvement CI lower bound > 0;
- direction Brier improvement CI lower bound > 0;
- availability does not regress;
- the `L=3 → H=30` anchor has at least 30 games and no sign reversal;
- all authority, fold, preprocessing, calibration, and source hashes verify.

The one-standard-error rule runs only among candidates that pass every gate.
If none pass, publish a content-addressed `NO_MODEL_ADVANCE` SelectionLock
binding the full matrix, panel, folds, thresholds, source hashes, and terminal
decision. That negative lock blocks Kalshi winner transport, semantic winner
ablation, magnitude follow-up, and any attempt to retune the same development
matrix.

## 12. Kalshi validation

Kalshi validates only a fully verified Polymarket `SelectionLock` that fixes
the model/block, L/H grid, target and direction threshold, folds, panel
lineage, and every producer/preprocessor/calibrator hash. A
`NO_MODEL_ADVANCE` lock cannot enter this step.

Two rule gates are kept separate:

1. `COMPLETED_NON_TIE_REALIZATION_COMPARABLE` proves the same completed,
   non-tied NFL game, home-team outcome, and realized label and permits only a
   descriptive frozen winner+B0 transport.
2. `STRICT_CONTRACT_RULE_EQUIVALENT` additionally proves frozen native
   settlement treatment for tie, postpone, cancel, abandon, and other edge
   cases. Only this gate permits formal transport-validation language.

The normalized contract canonical hash is not native settlement-rule proof.
Likewise, `transport_pair_id` proves row identity only; it cannot by itself
prove proposition or settlement equivalence.

Positive gate evidence must come from a separately verified,
content-addressed rule-audit artifact that binds the exact venue contract IDs,
captured native metadata object hashes, adjudicated rule fields, and
development-game cohort. Mutating an already verified in-memory Panel frame
cannot create authority. Until that artifact is bound and reloaded through the
formal input verifier, both gates remain `UNPROVEN` for the immutable exact-153
batch and Kalshi transport stays blocked.

When those gates permit the corresponding analysis:

- no reranking;
- no target recalibration;
- no feature or horizon changes;
- venue-independent `transport_pair_id` proves exact game, information anchor,
  winner proposition, home/focal outcome, L, H, and target-version identity;
- the denominator is every eligible anchor/L/H with a valid Kalshi winner
  binding; no post trade means `availability_h=0`, not a dropped row;
- transported winner versus transported B0;
- winner-minus-B0 uses exact paired Kalshi row IDs before
  row→anchor→game aggregation and game bootstrap;
- availability and direction reported separately;
- direction is evaluated only where an actual Kalshi trade is available;
- cross-venue comparison uses a separate common-support paired table;
- native Kalshi comparator is not a second selection round;
- at least 30 clean anchor games.

A Kalshi failure does not cause another Polymarket model to be selected. The
result remains at most `POLYMARKET_ONLY_CANDIDATE`. If only completed non-tie
comparability passes, the Kalshi output is explicitly descriptive and
`formal_transport_validation_allowed=false`.

Kalshi is a cross-venue transport stress test on the same 153 sports games,
not an independent sports holdout or execution test. It cannot establish that
Kalshi is slower, nor support BBO, lead-lag, or tradable-alpha claims.

## 13. Semantic-group ablation and shortlist

Only the frozen Polymarket winner receives drop-one-group OOF ablation:

```text
full D3
vs D3-action
vs D3-possession-result
vs D3-score-result
vs D3-kick-result
vs D3-adjudication
vs D3-reference/prestate
```

Each group reports:

- joint and direction loss delta;
- 10,000 game-cluster bootstrap CI;
- at least 30 games and 100 unique episodes for scientific promotion;
- LOO sign stability;
- maximum single-game contribution;
- family-level BH correction.

The lower 30-game/20-episode gate is only a computational minimum. Rare factors
remain `CASE_ONLY`.

## 14. Magnitude follow-up

Only after a direction winner is frozen:

- use the same folds, pairs, winner feature block, and D0;
- condition on realized UP and DOWN;
- estimate `|Δp|` q10/q25/q50/q75/q90;
- report pinball loss, approximate CRPS, interval coverage, anchor, and
  game-bootstrap stability.

Magnitude failure blocks magnitude claims only.

## 15. Holdout gate

The 81-game holdout remains:

```text
reaction_read_count = 0
holdout_reaction_accessed = false
```

It opens once, only after:

1. episode reconciliation passes;
2. regulation/PIT provenance passes;
3. all corrected 45 cells verify;
4. Polymarket selection is frozen;
5. Kalshi transport is reported;
6. semantic-group and sports-expert review is complete;
7. definitions, time grid, models, thresholds, and source hashes are sealed in
   `ShortlistLockV1`.

No tuning and rereading of the same holdout is allowed.

## 16. Current execution status

Status frozen on 2026-07-29:

```text
exact-45 = COMPLETE / PASS
selection = NO_MODEL_ADVANCE
winner = null
Kalshi candidate transport = NOT_APPLICABLE
formal contract transport = BLOCKED (strict rule equivalence 0/153)
holdout reaction reads = 0
```

The terminal negative lock is not an implementation failure.  It is the
pre-registered outcome when no learned candidate passes every mandatory gate.
Logistic D1-D3 improved integrated joint loss and availability but did not
show stable conditional-direction improvement.  XGBoost D3 improved
conditional direction while regressing availability and integrated joint
loss.  Therefore no candidate can be transported or tested on the sealed
holdout.

Completed:

- [x] Facts V4 exact-153 source reconciliation.
- [x] Stage A v1 model/calibration diagnostic.
- [x] PanelV2 and old exact-45 audit.
- [x] old run stopped after 27 diagnostic cells.
- [x] independent sports, statistics, provenance, and Kalshi reviews.
- [x] report hard-gates old diagnostics from shortlist/holdout.
- [x] publish and independently audit `FinalizedEpisodeV1`
      (`BLOCKED_DIAGNOSTIC` after semantic audit).
- [x] publish PIT-safe Stage A prestate context.
- [x] prove hash-equivalent Stage-B hot-path optimization.
- [x] publish and independently audit `RegulationDecisionPanelV3`
      (`BLOCKED_DIAGNOSTIC` because its episode input is blocked).
- [x] accept ADR 0007 from exact-153 evidence.
- [x] publish corrected `ProvisionalInformationAnchorV2` and
      `FinalizedEpisodeV2`;
- [x] publish one-to-one `EventPrestateContextV2`;
- [x] build and independently audit canonical
      `RegulationDecisionPanelV4`;
- [x] run and independently audit the corrected 45-cell Polymarket OOF
      matrix (854,676 accumulated cell rows; 94,964 unique OOF source rows,
      123 OOF games);
- [x] freeze and independently reproduce the Polymarket
      `NO_MODEL_ADVANCE` SelectionLock;
- [x] publish native-rule audit: completed non-tie comparability 153/153,
      strict rule equivalence 0/153;
- [x] bind the immutable native-rule overlay without mutating PanelV4:
      Panel trade/token lineage and native market/metadata lineage remain
      separate, cross-bound by `game_id + venue`;
- [x] mark Kalshi candidate transport `NOT_APPLICABLE` because the frozen
      winner is null, while separately blocking formal contract transport;
- [x] block semantic winner ablation and conditional magnitude follow-up
      because no model advanced;
- [x] keep the 81-game holdout sealed because the shortlist gate cannot open;
- [x] publish the canonical Chinese report, heatmaps, and Notebook section.
- [x] pass the final focused suite (94 tests), the real 153-game native-rule
      overlay verification, and independent model/report reviews;
- [x] publish this release on SAF `main`.

## 17. NO-GO

- No real money, order, maker, hedge, or cross-venue execution.
- No historical L2/BBO, OFI, depth, queue, or fill fabrication.
- No midpoint treated as executable P&L.
- No holdout read before lock.
- No unregistered factor or post-result horizon tuning.
- No README/leaderboard return claim as evidence.
- No LLM hot path, RL, or microservice expansion.
