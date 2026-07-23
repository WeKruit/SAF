# Program Status v0

- **Owner:** Team A
- **Version:** v0
- **Due gate:** `first_round_artifact_gate`
- **Decision:** `CONDITIONAL_GO`
- **Execution state:** `NO_REAL_MONEY`

The repository foundation, Team A contracts, Team H registry, immutable raw store, Team E label specification, Team F PRELIMINARY taker/TCA path, Team G audit workflow and reports, and Team I fail-closed matrices exist. Current checked-in evidence also includes the X-01 24-hour PMXT input preflight, the exact four-day X-02 input bundle, a bounded-incomplete Kalshi public REST prefix, the Polymarket recorder implementation and its current short observation, real-data X-11/X-12 POCs, deterministic NBA/NFL/soccer/MLB state reducers, registered NFL-drive and soccer-five-minute full-path latency measurements, and a fail-closed prediction-market alignment audit. The X-09 fixture harness passes replay Levels 1+2. This is an implementation status statement, not a promotion decision or proof that a registered experiment is complete.

## Experiment state

- X-01: `DATA_READY_PREFLIGHT_ONLY`; the selected 2026-05-28 UTC day has a frozen 24-hour, 24-object input manifest and preflight. The preflight explicitly records `reconstruction_executed: false`; all-contract full-day semantic reconstruction, independent price/size comparison, and remaining registration locks are open.
- X-02: `MEASUREMENT_COMPLETED_REGISTRY_ACCEPTANCE_DEPENDENCY_BLOCKED`; the preregistered four-day audit scanned 96 objects and 6,549,816,634 rows. P50 is 177 ms, P95 is 159,161 ms, absolute P99 is 279,725 ms, negative rate is zero, and canonical receive/source disorder is 0.019201353%. The registered `absolute_p99_ms_gt_5000` rule downgrades millisecond-sensitive research to seconds. The experiment card's `results_ref` remains empty and no promotion decision is claimed because X-01 is incomplete under RB-014.
- X-03: `POC_ONLY`; the fail-closed 28-day normalized-input census aggregator and tests exist, including PIT-missing exclusion and deterministic sport-by-venue metrics. No frozen four-week PMXT/Polymarket/Kalshi input manifest or empirical census result exists, so the formal experiment remains incomplete.
- X-04: registered; no formal event-study execution has started because X-01 and X-02 are incomplete.
- X-05: specification complete; label generation remains blocked by unresolved locks and X-01.
- X-06: model/validation POC and a 12-event deterministic synthetic state-reducer replay exist; empirical model accuracy and inference latency remain unmeasured because O-005 and the real-data choices are blocked.
- X-07: pipeline output is PRELIMINARY only; formal and go/no-go scopes remain blocked.
- X-08: the original 15-second, five-frame observation remains the initial access smoke only. The later alignment-audit cutoff verifies 17 sealed Polymarket manifests through UTC hour 15, and the repository now includes a separately hash-verified hour-17 manifest with 4,006 records. None is a formal X-08 result and the seven-day gate is not met; persistent operation still needs a durable host. Kalshi public REST evidence is `BOUNDED_INCOMPLETE_LICENSE_PENDING`: markets and trades each verify 100 pages and 100,000 unique records, but both terminal cursors are nonempty, O-003 is pending, historical L2 is not claimed, and live/historical overlap was not run.
- X-09: deterministic fixture status is `HARNESS_PASS`; formal experiment state is `EXPERIMENT_BLOCKED` until its locks and X-01 clear.
- X-10: workflow/reports complete; experiment state is `REGISTERED_NOT_RUN` because the 50-pair sample and review protocol are not locked.
- X-11: `PRELIMINARY_POC`; the real-data full run and five governed model bindings remain separate from reducer-v2 evidence. Reducer v2 fixes same-row score/timeout attribution and independently checks 181 transitions in one complete real game; the old 281/285 v1 scan is superseded and cannot be cited. Season-wide order, suspension/resume, clock correction and postseason OT rules remain open. Model method status remains `PIT_UNPROVEN`, the models remain `poc_only`, and the evidence is not eligible for a formal result.
- X-12: `POC_ONLY`; reducer v2 deterministically completes the frozen 380-match / 1,313,773-event season twice with zero final-score mismatches, while full registry-backed envelope integration remains a one-match test. O-004 remains research-only, event availability is offline rather than live PIT, no point-in-time market prior exists, and the current five-minute head is not state-conditioned. Formal promotion remains unauthorized.

MLB now has a source inventory and one complete-game deterministic reducer validation, but no registered probabilistic experiment or model. F1 remains a source inventory only. Neither sport has a calibration result or trading conclusion.

## Game-state and prediction-market alignment

- NBA: the source-observed reducer replays a registered 12-event synthetic X-06 fixture deterministically. Reducer-only p50/p95/p99 is 3,416/3,500/3,542 ns; real-game count and model-accuracy count are zero.
- NFL: reducer v2 independently checks one frozen 182-row game across 181 transitions twice, including same-row TD/XP and timeout attribution plus explicit context carry. Reducer-only p50/p95/p99 is 3,709/3,875/3,917 ns. The registered next-drive transition full path is 0.178333/0.181542/0.186917 ms and about 5,648 events/s; final-outcome logistic/GBDT are not yet measured through that same path. No current v2 season census or live SLA exists.
- Soccer: reducer v2 completes 380/380 frozen matches twice, reducing 1,313,773 events per run with zero final-score mismatches and identical aggregate hash. Reducer-only single-match p50/p95/p99 is 36,292/47,250/49,500 ns. The five-minute POC full path is 0.094667/0.097375/0.099792 ms and about 10,539 events/s, but that head repeats pregame intensities rather than using current state. This is not a live SLA.
- MLB: one frozen 88-event real game has 87 zero-mismatch next-observation comparisons and repeatable replay hash. Reducer-only p50/p95/p99, including full offline provenance continuity checks, is 17,042/17,709/18,000 ns. No probabilistic model exists.
- F1: no state reducer, replay, or probabilistic model exists.
- Prediction-market matched-as-of status is `NOT_ALIGNED`: canonical game-condition-outcome assertions, MarketMetadataSnapshotV0 instances, VenueRuleSnapshot instances, joinable ModelOutputV1 instances, and matched rows are all zero. Therefore no sport has a validated prediction-market symmetry, disagreement, lead-lag, mispricing, or alpha result.

## Binding NO-GO

Real-money trading, live maker, multi-venue simultaneous live arbitrage, live copy trading, LLM hot path, RL, large-scale microservices, unregistered quick backtests, and README-return evidence remain prohibited. PMXT L2 is used only for deterministic level-2 book reconstruction; it cannot establish queue position or exact queue fill.

## User resources required

The next external inputs are: Kalshi live-L2 API key ID and RSA private-key path; intended operating jurisdiction/entity/account context; a persistent recorder host with durable storage and monitoring; WORM/object-lock policy for production raw data; an approved independent X-01 price/size comparison source; durable storage and retention for the X-03 four-week sport census; licensed point-in-time sports data or written permissions; and an approved external secret manager inventory/rotation policy with an incident owner. Team H must separately resolve each experiment's remaining registered locks before formal empirical acceptance.
