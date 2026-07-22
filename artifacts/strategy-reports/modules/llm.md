# LLM Module Report v0

- **Owner:** Team G2
- **Version:** v0
- **Due gate:** `2026-08-05_W2_review`
- **Scope:** `NO_DEMO_NO_LIVE`; **slow path only**

## Evidence

R-043/R-044/R-047/R-048/R-049 cover benchmarking, belief-to-trade gaps, simple forecasting, and risk-manager claims; R-045/R-050 are frontier backlog. These require evidence cards before implementation claims.

## Real execution difficulty

Model knowledge time, retrieval latency, prompt/version drift, calibration, abstention, reproducibility, and token cost all separate forecast quality from executable value. An LLM may not enter the hot path.

## Honest backtest design

Pin model, prompt, retrieval corpus cutoff, tool outputs, and seed before evaluation. Score calibration against market prior, preserve abstentions and failures, then apply delayed executable-price costs in a separate registered experiment.
