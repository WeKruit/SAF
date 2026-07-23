# Soccer Evidence and Pipeline POC v0

- **Owner:** Team D3
- **Version:** v0
- **Due gate:** `2026-08-05_W2_review`; O-004 remains `Team_I_compliance_green`
- **Status:** `POC_NO_PIT_MARKET_PRIOR`

The classic POC is a low-score-adjusted pregame Dixon-Coles Poisson outcome distribution. It normalizes truncated home/draw/away mass and is evaluated with chronological expanding folds. The current five-minute head reuses the same pregame intensity pair at every cutoff; it is not conditioned on current score, remaining time, cards, possession, or reducer state. R-006/R-007/R-008 are the core evidence queue; I-019 is the StatsBomb POC; I-020 remains background only.

Team H records X-12 with `execution_authorized: true` for its POC scope. The repository contains a POC real-data X-12 result with 1X2 and five-minute Brier/log-loss/calibration evidence, plus a reducer-v2 census over all 380 frozen matches. Formal promotion remains unauthorized: StatsBomb is research-only, event publication is offline rather than live PIT, there is no PIT market prior, and registration locks remain unresolved. The next fitted model must first amend X-12 to freeze a state-conditioned Cox/Poisson feature set and disjoint calibration interval.
