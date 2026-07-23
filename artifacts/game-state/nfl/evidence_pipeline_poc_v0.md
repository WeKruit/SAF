# NFL Evidence and Pipeline POC v0

- **Owner:** Team D2
- **Version:** v0
- **Due gate:** `2026-08-05_W2_review`
- **Status:** `PRELIMINARY_POC_PIT_UNPROVEN`

The classic logistic state vector contains score differential, time remaining, possession, and both timeout counts. It is designed for the shared game-grouped walk-forward and calibration evaluator. R-014/R-015 are the core evidence cards; R-016 is advanced; I-018 is the nflfastR pipeline artifact.

Team H records X-11 with `execution_authorized: true` for its POC scope; a formal result remains unauthorized. The repository contains a PRELIMINARY real-data X-11 walk-forward result over the pinned 2015–2025 nflverse releases, including Brier, log loss, calibration slope/intercept and game-cluster bootstrap intervals. It remains `PIT_UNPROVEN`: the spread prior has no exact pregame observation timestamp, formal locks remain unresolved, and no prediction-market quote is joined. Reducer-v2 engineering evidence is separate from the fitted-model pipeline and does not promote the result.
