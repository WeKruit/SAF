# NFL Evidence and Pipeline POC v0

- **Owner:** Team D2
- **Version:** v0
- **Due gate:** `2026-08-05_W2_review`
- **Status:** `POC_COMPLETE_CALIBRATION_RESULT_NOT_AUTHORIZED`

The classic logistic state vector contains score differential, time remaining, possession, and both timeout counts. It is designed for the shared game-grouped walk-forward and calibration evaluator. R-014/R-015 are the core evidence cards; R-016 is advanced; I-018 is the nflfastR pipeline artifact.

No empirical calibration result is written. Team H registered the D2 work as X-11 with `execution_authorized: true` for its preregistered pipeline only; the formal result scope remains unauthorized, and its nflverse manifest, PIT feature, model/config/seed, bootstrap, tie-policy, and split locks remain unresolved. The next valid action is to resolve those exact X-11 registrations against the pinned nflverse release before running the registered evaluator.
