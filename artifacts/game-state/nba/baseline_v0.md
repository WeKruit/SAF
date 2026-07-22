# NBA Game-State Baseline v0

- **Owner:** Team D1 + Team H
- **Version:** v0
- **Experiment:** `X-06`
- **Due gate:** `2026-08-05_W2_review`
- **Status:** `PIPELINE_POC_COMPLETE_EMPIRICAL_RUN_BLOCKED`

The POC compares the point-in-time market prior, standardized logistic regression, and histogram GBDT on identical game-grouped chronological walk-forward folds. Every feature carries an availability timestamp no later than prediction time. Required reporting is Brier score, log loss, calibration slope/intercept, and game-cluster bootstrap 95% intervals.

No empirical NBA calibration values are reported. O-005 is `NOT_GREEN_BLOCKED`: current NBA.com terms prohibit NBA Statistics use with gambling absent permission, and no alternative licensed input manifest was supplied. X-06 data/prior/feature choices also remain registration locks. Synthetic unit tests validate code behavior but are not evidence.

Catalog evidence queue: R-009/R-010/R-017/R-018 core; R-011/R-012 background; R-013 advanced; I-021 pipeline; O-005 license gate.
