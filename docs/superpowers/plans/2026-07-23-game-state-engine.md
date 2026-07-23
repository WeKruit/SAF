# Game-State Engine Implementation Plan

Date: 2026-07-23

## Wave 1 — domain foundation

1. Add failing common-interface tests.
2. Implement immutable common state/event protocols, identity validation,
   deterministic transition traces, and canonical hashes.
3. Add a stage-separated latency benchmark with deterministic report format.

## Wave 2 — parallel sport reducers

1. NFL: play-level state, normalized observation adapter, reducer, invariants.
2. Soccer: StatsBomb event-level state, adapter, reducer, invariants.
3. MLB: parsed Retrosheet/Chadwick state, reducer, invariants.
4. NBA/F1: defer empirical code until their data-rights gates; do not fabricate
   results from fixtures.

## Wave 3 — preregistration and real-data validation

1. Amend X-11 and X-12 before inspecting new reducer accuracy or latency
   measurements.
2. Register the MLB empirical model experiment before running its model.
3. Freeze source objects, code/config hashes, supported-event rules, splits,
   seeds, bootstrap, and latency protocol.
4. Run bounded validation first; only then run full frozen data.

## Wave 4 — predictors

1. Reproduce nflfastR/Lock-Nettleton state baseline and existing next-drive
   model on the new state interface.
2. Replace the Soccer score-only constant-rate transition with a
   state-conditioned SPADL/VAEP or hazard baseline.
3. Reproduce the MLB 24-state Markov baseline after Chadwick validation.
4. Report Brier, log loss, calibration slope/intercept, game-cluster bootstrap
   confidence intervals, model-minus-prior paired confidence intervals, and
   stage latency.

## Wave 5 — prediction-market alignment

1. Build canonical sport-game-to-condition mapping from PIT metadata.
2. Join model states and market quotes as of the same event cutoff.
3. Report disagreement, lead/lag, spread/depth conditioning, and executable
   bid/ask feasibility.
4. Do not produce a trading or alpha claim unless the registered gate passes.

## Verification

- targeted tests after each module;
- complete repository tests;
- deterministic replay twice with identical hashes;
- registry/program audit;
- artifact self-hash and lineage audit;
- clean Git state and push the exact verified commit to SAF.
