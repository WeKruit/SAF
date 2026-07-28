# Primary-source review

Observed: 2026-07-25 UTC.

This review uses official project documentation, release tags, competition
pages, or the authors' own public submissions.  Published metrics, rankings,
and conclusions are recorded only to understand a method.  They are not
evidence for X-13.

## Mature frameworks

### nflfastR

- Source: [nflfastR v5.2.0](https://github.com/nflverse/nflfastR/tree/v5.2.0)
- Frozen identity:
  `v5.2.0@675a817a1563d9c4f18cbb127caecf6e9a33be25`
- Code license: MIT, independently checked at the tagged
  [LICENSE.md](https://github.com/nflverse/nflfastR/blob/675a817a1563d9c4f18cbb127caecf6e9a33be25/LICENSE.md).
- Data license: the local `DS-NFLVERSE` registration records CC-BY-4.0.
- SAF use: read the already-computed EP/EPA/WP/WPA/CP/CPOE/XYAC fields from
  immutable Parquet, orient them to the focal team, and expose them as
  diagnostic football context.
- Boundary: the source model's training claims are not revalidated merely
  because a column exists.  WP is not prediction-market fair value.

### fastrmodels

- Source: [official model archive](https://github.com/nflverse/fastrmodels/tree/model_archive)
- Frozen identity:
  `model_archive@9f2495fdb4943087ca663d96706eb5df7973aff4`
- Code/model license: MIT at the tagged
  [LICENSE.md](https://github.com/nflverse/fastrmodels/blob/9f2495fdb4943087ca663d96706eb5df7973aff4/LICENSE.md).
- SAF use: the frozen no-spread WP model is already governed as
  `DS-NFL-FASTRMODELS`; X-11 reproduction protocols V1/V2 are the independent
  local evidence.
- Boundary: diagnostic football probability only.  The archive does not
  establish event-time availability or a market-value target.

### nflreadpy

- Source: [nflreadpy v0.1.5](https://github.com/nflverse/nflreadpy/tree/v0.1.5)
- Frozen identity:
  `v0.1.5@95b6cb50852523d043b8bf3abc62e801a3654b7d`
- Code license: MIT at
  [LICENSE.md](https://github.com/nflverse/nflreadpy/blob/95b6cb50852523d043b8bf3abc62e801a3654b7d/LICENSE.md).
- SAF use: borrow its small loader surface, Polars orientation, and explicit
  memory/filesystem/off cache modes.
- Boundary: the package is not installed as a data source and notebooks may
  not use its network loaders.  SAF's object hash and manifest checks remain
  authoritative.

### xpass and pass_oe

- Source: the official
  [`add_xpass` documentation](https://github.com/nflverse/nflfastR/blob/675a817a1563d9c4f18cbb127caecf6e9a33be25/man/add_xpass.Rd)
  in nflfastR v5.2.0.
- Meaning: `xpass` is the expected dropback probability; `pass_oe` is the
  observed dropback choice relative to that expectation on the documented
  0–100 scale.
- SAF use: play-call surprise from the frozen columns.
- Boundary: it does not measure pass quality, yards surprise, pressure,
  turnover surprise, or market mispricing.

### nfl4th

- Source: [nfl4th v1.0.7](https://github.com/nflverse/nfl4th/tree/v1.0.7)
- Frozen identity:
  `v1.0.7@886c61a329be47f3bb294e1a5ab65806cf4db39f`
- Code license: MIT at
  [LICENSE.md](https://github.com/nflverse/nfl4th/blob/886c61a329be47f3bb294e1a5ab65806cf4db39f/LICENSE.md).
- SAF use: an isolated R build will emit go, field-goal, and punt
  action-conditional WP plus chosen-versus-best regret to hashed Parquet.
- Boundary: the official project documents edge cases for returns, players,
  penalties, weather, and blocked kicks.  No nfl4th factor is available until
  the pinned build and fixed fixtures pass.

### Quarto

- Source: official
  [parameter documentation](https://quarto.org/docs/computations/parameters.html)
  and [v1.9.38 release](https://github.com/quarto-dev/quarto-cli/releases/tag/v1.9.38).
- Frozen identity:
  `v1.9.38@6ebb5db80eb542ac76189c3d7c33ae6f654b93d2`
- macOS archive SHA-256:
  `47089a5020cfb41981ba0d4b46e110edfa608722aea45ef248e14efba6d6b18a`
- License: MIT at
  [COPYING.md](https://github.com/quarto-dev/quarto-cli/blob/v1.9.38/COPYING.md).
- SAF use: parameterized, offline Jupyter reports.
- Boundary: a renderer cannot define a factor, override a registry predicate,
  or turn notebook output into a formal result.

## Big Data Bowl program references

- [Big Data Bowl 2025](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025):
  method reference for pre-snap structure, expected-versus-observed design,
  and play-specific explanation.
- [Big Data Bowl 2026 Analytics](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics):
  method reference for spatial/player representations while the ball is in
  the air.
- [NFL Football Operations Big Data Bowl](https://operations.nfl.com/gameday/analytics/big-data-bowl/):
  official description of the contest and its use of Next Gen Stats.

Competition rules and the license of each submission are separate questions.
SAF has not registered the competition tracking bytes, so tracking-dependent
features remain `DATA_GAP`.

## Author submissions

### Expected Field Position on Punts

- Author source:
  [Jack Lichtenstein's notebook](https://www.kaggle.com/code/jacklichtenstein/expected-field-position-on-punts)
- Frozen notebook identity: `script_version_id:84061270`.
- Method structure: model five punt outcomes and expected post-punt field
  position, then compare expected with observed field position or EPA.  The
  notebook describes 2018–2019 model development, a 2020 season holdout, and
  five-fold outcome-stratified tuning by multiclass log loss.
- Portable idea: punt is not a single routine category; preserve return,
  fair-catch, out-of-bounds, touchback, and downed outcomes.
- Blocker: tracking and PFF charting are absent, and notebook/data licenses
  were not independently verified.  No code, data, model score, or ranking is
  imported.

### Pressures Over Predicted

- Author source:
  [Steven Patton's notebook](https://www.kaggle.com/code/stevenpatton97/pressures-over-predicted-pop)
- Frozen notebook identity: `script_version_id:115938242`.
- Method structure: estimate context-conditioned binary pressure probability
  from tracking and pocket/leverage features, then compare observed pressure
  with expected pressure.  The notebook reports ten-fold cross-validation and
  F1, but SAF has not verified a game-grouped split.
- Portable idea: expected-versus-observed residuals must use out-of-sample
  probabilities.
- Blocker: X-13 lacks tracking, pocket geometry, blocker leverage, and a
  complete PIT pressure label.  No `pressure_over_expected` factor is created.

### Safety Entropy

- Author source:
  [Cole Jacobson and collaborators' notebook](https://www.kaggle.com/code/colejacobson/safety-entropy)
- Frozen notebook identity: `script_version_id:216371603`.
- Method structure: estimate MOFO probability from pre-snap safety movement
  and summarize uncertainty with entropy.  The notebook describes separate
  one- and two-safety neural networks, ten-fold comparison against logistic
  baselines, and bootstrap intervals.
- Portable idea: defensive structure and predictive uncertainty can moderate
  an event factor, and aggregate results should retain concrete play review.
- Blocker: X-13 lacks safety tracking and MOFO/MOFC labels.  Notebook and
  linked-code licenses were not verified.

### High Quality Interception Opportunity

- Author source:
  [HQIO writeup](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/writeups/HQIOs)
- Frozen identity: `kaggle_writeup_id:63159`.
- Verified scope: the author describes defender-created interception
  opportunity at ball arrival rather than only recorded interceptions.
- Portable idea: distinguish rare-event opportunity quality from realized
  outcome.
- Blocker: exact executable method, validation, code license, data license,
  and required tracking are not independently available.  X-13 therefore
  keeps turnover as observed taxonomy plus realized WP shock and explicitly
  forbids `turnover_over_expected`.

