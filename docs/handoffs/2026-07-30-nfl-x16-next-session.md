# NFL X-16 Next-Session Handoff

## Authoritative X-16 Plan Location

The complete, authoritative X-16 implementation plan is:

```text
Repository-relative:
docs/superpowers/plans/2026-07-30-nfl-x16-reaction-trajectory.md

Local absolute path:
/Users/wekruitclaw1/Desktop/prediction-market/.worktrees/hft-time-evidence-audit-v1/docs/superpowers/plans/2026-07-30-nfl-x16-reaction-trajectory.md

GitHub:
https://github.com/WeKruit/SAF/blob/codex/hft-time-evidence-audit-v1/docs/superpowers/plans/2026-07-30-nfl-x16-reaction-trajectory.md
```

This handoff is only the next-session startup prompt. It does not replace,
summarize, or override the X-16 plan. The next session must read the complete
plan before inspecting implementation files or delegating work.

Copy everything below the divider into a new Codex session.

---

You are continuing the SAF repository's NFL research program.

GitHub repository:

```text
https://github.com/WeKruit/SAF
```

Required branch:

```text
codex/hft-time-evidence-audit-v1
```

Local repository:

```text
/Users/wekruitclaw1/Desktop/prediction-market
```

Canonical implementation worktree:

```text
/Users/wekruitclaw1/Desktop/prediction-market/.worktrees/hft-time-evidence-audit-v1
```

First checkout and verify the required branch:

```bash
git fetch origin
git checkout codex/hft-time-evidence-audit-v1
git pull --ff-only origin codex/hft-time-evidence-audit-v1
git status --short --branch
git rev-parse HEAD
```

The authoritative X-16 plan is:

```text
docs/superpowers/plans/2026-07-30-nfl-x16-reaction-trajectory.md
```

Read that complete file first. Then read the three program-level sources of
truth before editing:

```text
charter/research_program_charter_v0.2.md
charter/catalog_registry.csv
charter/catalog_team_assignments.csv
```

Task:

> Implement and publish the complete X-16 NFL source-anchor-relative
> probability trajectory study. Start from the existing 153-game immutable
> sports facts and actual-trade sources. Build the missing decision-time
> feature, reference-value, exact-window target, OOF model, replication,
> Notebook, and report layers. Do not redesign the target, reopen X-15, or read
> the 81-game holdout before a passing development model and immutable
> selection lock exist.

The research question is:

```text
At an NFL information anchor [S,T], using only information known by T:

1. How much does the frozen nflfastR reference win probability change?
2. Is an actual home-contract trade observed in each registered future window?
3. Conditional on an actual trade, what is the 3-90 second probability path
   relative to the pre-anchor actual-trade VWAP p0?
4. Does adding sports state/event/reference information improve that
   conditional distribution over a strong market-only baseline?
```

Current facts:

- X-15 is complete and terminal `NO_MODEL_ADVANCE`; preserve all X-15 objects.
- X-16 sports-fact extraction and the 153-game fact-review layer already exist.
- X-16 `reaction-trajectory-v1` has been designed but is not implemented.
- Development contains 153 games.
- Holdout contains 81 sealed games; reaction read count must remain zero.
- PIA V2 contains 25,250 anchors and 25,070 primary-supported anchors.
- Market inputs contain 4,773,299 actual observations.
- Kalshi is primary; Polymarket is frozen replication only.
- No upstream download is needed.
- Existing `.env.s3` contains storage credentials; never print or commit them.
- Missing local immutable objects may be hydrated only by manifest SHA from the
  existing Supabase S3 store.

Do not confuse these states:

| Layer | Current status | Next-session action |
|---|---|---|
| 153-game canonical sports replay/facts | Complete and reusable | Verify manifests; do not replay upstream |
| Exact-153 X-16 facts V3 | Complete and reusable | Read through its batch manifest |
| X-15 historical actual-trade development panel | Complete and reusable as source evidence | Verify and project native trades; do not inherit X-15 labels |
| X-15 `mark_L/mark_H`, direction labels, and OOF models | Terminal historical evidence | Never use as X-16 target or compatibility schema |
| X-16 decision-time feature snapshots | Missing | Build new canonical artifact |
| X-16 nflfastR reference observations | Missing | Build with field-level PIT provenance |
| X-16 3-90 second VWAP trajectory targets | Missing | Build per game from actual trades |
| X-16 B0/M1 chronological OOF | Missing | Run only after reconciliation publication |
| X-16 Polymarket frozen replication | Missing | Run only after Kalshi definitions lock |
| X-16 selection lock | Missing | Create only if every development gate passes |
| 81-game holdout result | Sealed and unread | Consume once only after a valid passing lock |

Start by reading these reusable implementations and authorities:

```text
src/prediction_market/sports/nfl_x16_fact_extraction.py
src/prediction_market/sports/nfl_x16_fact_publication.py
src/prediction_market/sports/nfl_x16_exact153.py
src/prediction_market/research/nfl_x15_finalized_episodes.py
src/prediction_market/research/nfl_x15_development_panel.py
src/prediction_market/research/nfl_x15_native_rule_audit.py
src/prediction_market/research/nfl_x15_stage_a_support_audit.py

artifacts/market-observation/nfl/x16/exact-153-facts-v3/
artifacts/market-observation/nfl/x15/finalized-episodes-regulation-v2/
artifacts/market-observation/nfl/x15/historical-trades-only-development-panel-v2/
artifacts/market-observation/nfl/x15/stage-a-decision-support-audit-v1/

notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb
```

Important frozen manifest entry points:

```text
X-16 exact-153 facts:
artifacts/market-observation/nfl/x16/exact-153-facts-v3/batches/manifests/sha256/f8/f8713c13b9e5700bbfda9cfff87f712726b6b538754103afa61c3e600cec9f6b.batch-index.json

PIA/finalized episodes V2:
artifacts/market-observation/nfl/x15/finalized-episodes-regulation-v2/manifests/sha256/fa/fac9db018900fae919fd94fd2fec064f5eb3fcae24255fbedcc913099ad2b208.manifest.json

X-15 development panel V2 batch namespace:
artifacts/market-observation/nfl/x15/historical-trades-only-development-panel-v2/batches/manifests/

Stage A support audit:
artifacts/market-observation/nfl/x15/stage-a-decision-support-audit-v1/manifests/sha256/17/17593d410b93a6eb8d21f2deedd8d78d1b566bb9ec41e51a4870ec4acedee219.stage-a-support-audit.json
```

Treat those paths as immutable input authorities, not schemas to extend.
Publish X-16 only under:

```text
artifacts/market-observation/nfl/x16/reaction-trajectory-v1/
```

Canonical X-16 time and price contract:

```text
ProvisionalInformationAnchorV2:
  [S,T] = [source_interval_start, source_interval_end]

H:
  3,5,10,15,20,25,30,35,40,45,50,55,60,75,90 seconds

p0:
  actual native home-outcome contract size-weighted VWAP in [S-3s,S)

p_h:
  actual native home-outcome contract size-weighted VWAP in
  (T+h-3s,T+h]

outputs:
  delta_probability_0_h = p_h - p0
  delta_pp_0_h          = 100 * (p_h - p0)
  delta_logit_0_h       = logit(p_h) - logit(p0)
```

Hard price rules:

- Use all valid actual trades inside the full window.
- Use exact Decimal arithmetic.
- Require one exact native home contract, `price in [0,1]`, `size > 0`, complete
  pagination proof, and unique native trade ID.
- Never substitute the last trade, candle, midpoint, away complement, derived
  complement, or forward-filled price.
- A zero/one price keeps raw probability and pp; logit is null and is not
  clipped.
- A second-resolution trade enters a window only if its entire source-time
  interval lies inside that window.
- Boundary overlap becomes `TIMESTAMP_MEMBERSHIP_AMBIGUOUS`.
- Publish all fixed anchor shifts
  `-10,-5,-3,-2,-1,0,+1,+2,+3,+5,+10`; never select a favorable shift.

Required missingness states:

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

Historical data does not prove venue availability. Always retain:

```text
historical_market_open_status = UNKNOWN
historical_suspension_status = UNKNOWN_NO_AUTHORITATIVE_CHANNEL
historical_tradeability_status = UNKNOWN
```

Target roles:

- `OPERATIONAL_EX_ANTE_TARGET` is the only formal model and promotion target.
  It predicts at T and retains future paths even if another sports event later
  occurs. Future-event audit fields must be physically absent from the model
  projection.
- `RETROSPECTIVE_ISOLATED_DIAGNOSTIC` censors a horizon when
  `next_pia.source_interval_start <= T+h`. It uses future absence of an event,
  so it is diagnostic only.
- Continuation uses `L={3,5,10}` and every registered H with `H-L>=3`,
  producing exactly 38 pairs. It is diagnostic only.

Feature-layer separation:

1. Human-review facts retain full finalized replay and reconciliation.
2. `NFLX16InformationFeatureSnapshotV1` contains only fields proved
   `known_at <= T`.

Each model feature must carry value, known-at time, source row, source hash,
provisional/final status, model-allowed flag, and exclusion reason.

Allowed formal inputs:

- continuous game seconds remaining, quarter, score margin;
- possession, down, distance, yardline, red zone, timeouts;
- provisional event family/subtype and event magnitude;
- provisional beneficiary;
- p0 and pre-anchor actual trade activity;
- Stage A reference values only after their PIT gate passes.

Explicitly excluded from the initial X-16 formal model:

- player/team fixed effects and player identity;
- formation/personnel tactical learning;
- rolling strength and season aggregate features;
- news or inferred injury severity;
- tracking, routes, coverage;
- final outcome and every post-T sports/market field.

Stage A contract:

- Use the already frozen nflfastR/fastrmodels no-spread XGBoost bytes.
- Do not download, retrain, or replace the reference model.
- Build `reference_before_anchor`, `reference_at_anchor`,
  `reference_delta_home`, and `reference_delta_beneficiary`.
- `reference_at_anchor` may use only provisional state known by T.
- Do not fill from the next play or final result.
- OT is `MODEL_SUPPORT_UNPROVEN`.
- Preserve model version, model-bytes SHA, input row hashes, field-level
  known-at provenance, and support status.

Stage B has two separate questions:

```text
Head 1:
P(actual trade observed in the horizon window |
  p0 observed, verified capture)

Head 2, conditional on an actual endpoint trade:
distribution of p_h and delta from p0
```

Never call Head 1 liquidity, availability, or “whether somebody bought.”

Formal comparison:

- `B0_market`: horizon, logit(p0), pre-anchor trade count/size, last-trade
  age, 30/60-second activity, timestamp resolution, and game time.
- `M1_full`: B0 plus decision-time sports state, provisional event semantics
  and magnitude, provisional beneficiary, and PIT-proven nflfastR reference.

Model families are frozen:

- observed-trade propensity: elastic-net logistic, `l1_ratio=0.5`,
  `C={0.01,0.1,1,10}`;
- direction/materiality: secondary elastic-net binary/multinomial models;
- conditional trajectory distribution: pooled-horizon L1 quantile regression
  at `q={0.10,0.25,0.50,0.75,0.90}`,
  `alpha={1e-4,1e-3,1e-2,1e-1}`.

Use blocked inner CV and the one-SE rule. Fit one pooled-horizon model, not
separate factor-by-horizon models. Save raw quantiles and separately publish
monotone rearrangement. Convert logit-delta quantiles back through
`expit(logit(p0)+q)` before calculating probability points.

Chronological outer validation is frozen:

| Fold | Train | Validate |
|---|---|---|
| F1 | Week 1-4 | Week 5-6 |
| F2 | Week 1-6 | Week 7-8 |
| F3 | Week 1-8 | Week 9-10 |
| F4 | Week 1-10 | Week 11-12 |

Keep one game, both venues, every anchor, and every horizon in one fold.
Preprocessing, imputation, hyperparameter selection, and quantile calibration
must be fitted inside outer-train. Weight losses:

```text
horizon within anchor -> anchor within game -> equal-weight game
```

Primary conditional-distribution metric:

```text
Q3 = mean pinball loss for q25, q50, q75
```

Development advancement requires every condition:

1. Kalshi operational paired Q3 improvement over B0 is at least 2%.
2. The one-sided 95% paired game-bootstrap lower bound is positive.
3. Observed-trade log-loss skill lower bound is at least -1%.
4. q25/q50/q75 empirical hit-rate errors are each at most 0.05.
5. 50% interval coverage error is at most 0.05 and its CI contains 0.50.
6. At least 3/4 folds have non-negative Q3 skill.
7. At least 10/15 horizons have non-negative Q3 skill.
8. No supported horizon is worse than B0 by more than 10%.
9. Leave-one-game-out total Q3 skill remains positive for every deletion.
10. Every source, PIT, window, orientation, reconciliation, and holdout-read
    gate hard-passes.

If any condition fails:

```text
NO_INCREMENTAL_MODEL_ADVANCE
```

Direction accuracy, isolated/continuation results, or Polymarket replication
cannot rescue a failed Kalshi operational model.

Execution order:

1. Preflight the branch, program-level sources, immutable manifests, disk/RSS,
   and holdout access counter. Record the exact starting commit.
2. Register X-16 and freeze source/hash/holdout authorities.
3. Implement the new contracts and holdout guard with focused failing tests.
4. In parallel, build:
   - decision-time feature snapshots;
   - Stage A nflfastR reference provenance;
   - exact-window trajectory targets one game at a time.
5. Publish exact reconciliation and attrition before fitting any model:
   - 25,250 total anchors;
   - 25,070 primary-supported anchors;
   - 752,100 horizon candidates;
   - 1,905,320 continuation candidates.
6. Run two identical exact-153 builds and require the same content hash.
7. Run Kalshi B0/M1 rolling-origin OOF models and all advancement gates.
8. Lock all feature, target, horizon, model, calibration, metric, and wording
   definitions before Polymarket replication.
9. Run Polymarket once with the frozen Kalshi definition; never retune on it.
10. Update the Master Notebook and publish the offline Chinese report from
    published manifests only.
11. Open the 81-game holdout once only if every development gate passes;
   otherwise publish `NO_INCREMENTAL_MODEL_ADVANCE` and leave it sealed.

Parallel ownership is allowed and recommended:

- Agent A owns only X-16 governance, registry, contracts, source authority,
  and holdout guard.
- Agent B owns only decision-time feature snapshots and Stage A reference
  provenance.
- Agent C owns only market trajectory targets, attrition, and deterministic
  single-game/batch publication.
- Main agent owns model integration, exact-153 orchestration,
  Polymarket replication, Notebook/report, and final verification.

Agents are not alone in the codebase. Give each agent explicit file ownership,
tell them not to revert other work, and integrate only after targeted tests.

Implementation rules:

- Write focused failing tests before production code.
- Do not create a compatibility wrapper around PanelV4.
- Do not reuse old `mark_L/mark_H`, old direction labels, exact-45 selection
  results, or cumulative X-13 reaction paths.
- Formal model features must satisfy `known_at <= T`.
- Use the actual native home contract only.
- No trade means missing/observed-trade propensity, never zero/no-move.
- Operational is the only formal prediction target.
- Isolated and continuation are retrospective diagnostics only.
- Do not pool venues or treat two venue rows as independent games.
- All inference, bootstrap, folds, and losses are game-clustered/equal-game.
- No L2, BBO, execution, P&L, Sharpe, causality, latency, or alpha claims.

Required checkpoints:

1. Exact p0/p_h window-boundary and Decimal VWAP fixtures pass.
2. Empty windows remain null and are distinct from capture gaps.
3. Operational features physically exclude every future-event field.
4. `next_start == T+h` censors isolated analysis.
5. Exactly 38 continuation pairs are generated and reuse the same mark hashes.
6. Native home-contract identity and native trade dedupe hard-pass.
7. Future event, future trade, and final-result mutation do not change feature
   hashes or predictions.
8. Development candidate counts match exactly.
9. Two exact-153 builds produce the same content hash.
10. Same-game rows and both venues never cross folds.
11. Equal-game weighting is invariant to duplicate anchor rows.
12. Quantile ordering, hit rates, and interval coverage are published.
13. Holdout reaction access counter remains zero before a passing lock.
14. Notebook/report consume only published manifests and never invoke raw
    loaders or model fitters.
15. Final report prominently includes this claim boundary:

```text
PRELIMINARY_SOURCE_TIME_TRADES_ONLY
NON_CAUSAL
NON_EXECUTABLE
NON_ALPHA
```

Required X-16 public interfaces:

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

Required canonical artifacts:

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

Notebook/report requirements:

- Keep one authoritative Notebook:
  `notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb`.
- Notebook and HTML read content-addressed artifacts only.
- Show p0/p_h attrition, observed-trade propensity, conditional quantile
  heatmaps, games/anchors/support per cell, continuous time/state breakdown,
  Stage A provenance, B0 versus M1 OOF, calibration, game bootstrap, LOO,
  placebos, pseudo-anchors, clock shifts, Polymarket frozen replication, and
  concrete positive/negative/large/clock-sensitive/disagreement cases.
- Do not assemble a time path from changing survivor cohorts. A progressive
  or reversal claim requires a common complete-case risk set.

When complete, provide:

- commit and branch;
- exact artifact/report/notebook paths;
- development gate result;
- whether holdout remained sealed or was consumed;
- targeted test commands and full pass/fail counts;
- runtime and peak RSS;
- remaining data gaps and claim boundaries.

The final report must contain this exact substantive warning:

> This study estimates historical home-contract actual-trade VWAP paths
> relative to source-reported NFL information anchors and estimates price
> distributions only when an actual trade occurred in the stated three-second
> window. It does not estimate a latent price when no trade occurred,
> continuous venue availability, causal reaction latency, executable returns,
> or alpha. `p0` is a historical pre-anchor baseline, not a post-information
> executable entry price.

Do not stop after scaffolding. Continue through the complete 153-game
development publication, Stage A, Stage B, replication, report, and—only if
authorized by the locked gates—the one-time holdout.
