# NFL Official No-Spread Win-Probability Reproduction Plan

> **Execution rule:** No 2021–2025 prediction or metric may run before the
> Team H registration amendment is committed and validates. Asset loading on a
> synthetic vector is an engineering smoke only.

**Goal:** Reproduce the official nflverse `fastrmodels` no-spread win-
probability booster on frozen 2021–2025 nflverse rows, then publish
game-grouped accuracy, calibration, source-comparator diagnostics, and current
full-path latency without claiming live PIT or prediction-market alpha.

**Non-goals:** Do not run the spread booster, retrain the official model, fit a
calibrator, treat nflverse `home_wp` as ground truth, or change the existing
X-11 logistic/GBDT evidence.

## Frozen sources

- Feature/training specification commit:
  `75c7b68bc49535370236c38c9826265da075bd71`
- nflfastR feature helper commit:
  `ead5e2f9641490f692d923c04835bd3b90275b4e`
- Official model archive tag commit:
  `9f2495fdb4943087ca663d96706eb5df7973aff4`
- Official release asset ID: `253928623`
- Asset URL:
  `https://github.com/nflverse/fastrmodels/releases/download/model_archive/wp_model.ubj`
- Asset byte length: `106951`
- Asset SHA-256:
  `ff58c98d22e3e36f50f6e82f8e68d3ed0920101132a50d30bb70291df33c0a4c`
- Python runtime: `xgboost==3.3.0`
- Local native dependency: Homebrew `libomp` (runtime dependency only)

The official release already provides UBJSON. Do not add an RDA conversion
path.

## Protocol

### Exact feature order

1. `receive_2h_ko`
2. `home`
3. `half_seconds_remaining`
4. `game_seconds_remaining`
5. `Diff_Time_Ratio`
6. `score_differential`
7. `down`
8. `ydstogo`
9. `yardline_100`
10. `posteam_timeouts_remaining`
11. `defteam_timeouts_remaining`

`Diff_Time_Ratio` is:

```text
score_differential /
exp(-4 * ((3600 - game_seconds_remaining) / 3600))
```

The booster emits possession-team win probability. Convert it to home-team
probability with:

```text
p_home = p_posteam                 if posteam == home_team
p_home = 1 - p_posteam             otherwise
```

### Exact evaluation rows

- Frozen seasons 2021–2025, REG and POST.
- Regulation periods only: `qtr in {1,2,3,4}`.
- Exclude tied final games from binary metrics; report them separately.
- Require nonmissing `ep`, `score_differential`, `play_type`, `posteam`,
  `down`, `ydstogo`, `yardline_100`, `half_seconds_remaining`,
  `game_seconds_remaining`, `posteam_timeouts_remaining`, and
  `defteam_timeouts_remaining`.
- Require `posteam` and `defteam` to be game participants.
- Use frozen raw record ordinal as the final ordering tie-break. Never resolve
  the two known 2021 duplicate `order_sequence` values by silently dropping a
  row.
- Outcome labels come from the game result, never from `home_wp`.
- Source `home_wp` is diagnostic only because its transformer/model lineage is
  not pinned to the official asset above.

Expected pre-registered input census:

| Season | Raw rows | Games | Eligible rows | Eligible non-tie games |
|---|---:|---:|---:|---:|
| 2021 | 49,922 | 285 | 41,342 | 284 |
| 2022 | 49,434 | 284 | 40,937 | 282 |
| 2023 | 49,665 | 285 | 41,698 | 285 |
| 2024 | 49,492 | 285 | 41,269 | 285 |
| 2025 | 48,771 | 285 | 40,361 | 284 |
| Total | 247,284 | 1,424 | 205,607 | 1,420 |

This census is a filter/schema audit, not a prediction result.

### Metrics

- Row-micro and game-macro Brier score and log loss.
- Calibration slope/intercept and fixed reliability bins.
- Game-cluster bootstrap 95% confidence intervals, 200 resamples, seed
  `20260723`, reporting requested and valid resamples.
- Per-season and aggregate metrics.
- Absolute delta to source `home_wp` as a diagnostic distribution, never as an
  accuracy oracle.

No calibration model is fitted. Any future Platt, temperature, or isotonic
calibrator is a new model and requires a new preregistration.

## Task 1: Asset, registry, and Team H registration

**Files:**

- Modify `pyproject.toml` and `uv.lock`.
- Add one `DS-NFL-FASTRMODELS` row to `registries/dataset_registry.csv`,
  bound to catalog item `I-018`, MIT, approved, X-11 only.
- Preserve the exact UBJ under immutable `var/raw` and publish only its static
  manifest.
- Add `MODEL-NFL-FASTRMODELS-NO-SPREAD` v0 to
  `registries/model_registry.csv`.
- Cleanly change the X-11 reproduction contract from the old two-model
  placeholder to no-spread only; do not retain a compatibility alias.
- Add one append-only X-11 Team H registration amendment and ledger row.
- Add a checked-in protocol record containing every rule in this plan and bind
  its SHA in the registration.

**RED tests:**

- Wrong asset ID, byte length, SHA, tag commit, model feature count, or booster
  round count fails closed.
- The reproduction contract rejects spread model IDs.
- The reproduction amendment rejects evaluation before its timestamp.
- Code, data, model row, protocol, filter, bootstrap, or asset changes break
  the registration binding.
- Unresolved spread-prior locks do not apply to this no-spread reproduction;
  its own required locks are exact and independent.

**Acceptance:** Governance validates before any real prediction is executed.

## Task 2: Official predictor

**Files:**

- Create `src/prediction_market/models/nfl_fastrmodels.py`.
- Create `tests/models/test_nfl_fastrmodels_asset.py`.

**RED tests:**

- Missing/tampered manifest or model bytes.
- Wrong XGBoost runtime version.
- Wrong number or order of features.
- NaN/inf or ineligible state.
- Home/away orientation mutation.
- Synthetic golden vector expected prediction
  `0.84001481533050537` within a fixed numeric tolerance.

**Implementation:**

- Read through `read_verified_static_object`.
- Verify all frozen source identity fields before `Booster.load_model`.
- Use a preloaded immutable predictor for batch and single-row calls.
- Record that the golden vector is a runtime regression fixture, not an
  independent accuracy oracle.

## Task 3: Frozen evaluation and latency

**Files:**

- Create `src/prediction_market/sports/x11_fastrmodels.py`.
- Create `tests/sports/test_x11_fastrmodels.py`.
- Extend `src/prediction_market/sports/model_latency.py` only after the
  reducer-v3 season census passes.

**RED tests:**

- Exact eligible-row counts by year.
- Ties never enter binary metrics.
- Full games stay in one cluster.
- Labels do not read `home_wp`.
- `receive_2h_ko` follows the pinned helper definition.
- Duplicate source order is retained with raw ordinal tie-break.
- Two complete runs are byte-identical.
- Bootstrap requested/valid counts and seed are exact.
- Full-path timing includes state/event, feature projection, official booster,
  probability output validation; it excludes I/O, network, registry loading,
  and market joins.

**Execution gate:** Load the effective X-11 card and validate the exact
registered reproduction bindings immediately before reading evaluation rows.

**Artifacts:**

- `artifacts/game-state/nfl/fastrmodels_no_spread_reproduction_v1.json`
- `artifacts/game-state/nfl/fastrmodels_no_spread_reproduction_v1.md`

Both artifacts must say:

- `PRELIMINARY`
- `PIT_UNPROVEN`
- `offline_reconstruction_not_live_PIT`
- no fitted calibrator
- no prediction-market alignment
- no alpha evidence

## Task 4: Independent review and publication

- Spec review: prove that no spread field, future probability, README metric,
  or source `home_wp` enters prediction or labels.
- Quality review: reject broad exception handling, unverified direct file
  reads, hidden network access, or a model singleton that can mutate.
- Run focused tests, governance, then the full suite in a bounded temp root.
- Update the Chinese truth report only from the generated artifacts.
- Fast-forward to main and push SAF only after reviews and exact local/remote
  commit verification.
