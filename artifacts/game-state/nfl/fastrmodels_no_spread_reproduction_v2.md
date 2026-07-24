# NFL official no-spread reproduction v2

- Status: **PRELIMINARY**
- PIT status: **PIT_UNPROVEN**
- Observation mode: `offline_reconstruction_not_live_PIT`
- Calibrator: none fitted
- Prediction-market alignment: none
- Prediction-market symmetry: not evaluated
- Alpha evidence: none

## Frozen evaluation

- Eligible rows: 205607
- Eligible non-tie games: 1420
- Row-micro Brier: 0.166284941293
- Row-micro log loss: 0.491426483873
- Calibration slope: 0.987612141958
- Calibration intercept: -0.047990070225
- Game-macro Brier: 0.165508694357
- Game-macro log loss: 0.489080509097

## Current full-path latency

- Scope: event construction → state transition → official feature projection → preloaded booster → probability validation
- p50: 468896 ns
- p95: 1611483 ns
- p99: 4412516 ns
- Excludes I/O, registry loading, network, market joins, and any market/alpha interpretation.
