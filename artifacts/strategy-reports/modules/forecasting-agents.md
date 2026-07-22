# Forecasting-Agent Module Report v0

- **Owner:** Team G2
- **Version:** v0
- **Due gate:** `2026-08-19_W4_review`
- **Scope:** `NO_DEMO_NO_LIVE`

## Evidence

R-046–R-050 and I-014–I-017/I-025 cover multi-agent oracles, forecast benchmarks, PolyBench, Predict Raven, and agent tooling. Catalog status ranges from core evidence to frontier backlog; none is accepted as trading proof.

## Real execution difficulty

Forecast generation, aggregation, calibration, market mapping, order timing, risk constraints, and settlement are separate systems. Agent agreement can be correlated error rather than independent evidence.

## Honest backtest design

Freeze agent versions and information cutoff, prevent inter-agent leakage, compare single/simple/ensemble baselines, score calibration and abstention, and only then evaluate a separately registered delayed taker translation. Agents and LLMs remain off the hot path.
