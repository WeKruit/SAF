# Game-State Calibration and Public-Baseline Review v1

Date: 2026-07-23
Owner: Team H + D1 + D2 + D3 + D4 + D5
Status: primary-source review complete; proposed reproductions not yet run
Decision class: research-only, no alpha or trading conclusion

## Evidence classes

This review separates four evidence classes:

1. `SAF_REPRODUCED`: run from byte-exact governed inputs by this repository.
2. `PAPER_REPORTED`: reported by the cited authors and not reproduced by SAF.
3. `PROPOSED_REPRODUCTION`: technically and legally eligible, but it must be
   preregistered or amended before the first empirical run.
4. `BLOCKED_OR_INVENTORY_ONLY`: no empirical model claim is authorized.

Paper metrics, repository README results, and released model weights never become
SAF evidence without an independent governed reproduction.

## Current SAF facts

| Sport | Current governed result | What it proves | What it does not prove |
|---|---|---|---|
| NFL | `SAF_REPRODUCED`: 2015-2025 nflverse model POC; 2020-2025 per-game chronological walk-forward; reducer-v2 P0 replay on one full game | The separate model pipeline, clustered bootstrap, latency harness, and targeted same-row score/timeout reducer semantics execute deterministically | Reducer-v2 season completeness; exact live PIT availability of the spread; alpha; superiority to the official retrained fastrmodels feature specification |
| Soccer | `SAF_REPRODUCED`: StatsBomb Premier League 2015/16 research-only POC; reducer-v2 380-match census; expanding-window Dixon-Coles; five-minute transition output | Two reducer runs each complete 380/380 matches and 1,313,773 events with zero final-score mismatches and one aggregate hash; chronological fit/evaluation and latency harness execute | The current five-minute probabilities are not state-conditioned: the same pregame Dixon-Coles intensity pair is reused at every cutoff; the census has no independent event-by-event oracle and there is no PIT market prior |
| NBA | `BLOCKED_OR_INVENTORY_ONLY`: synthetic contract/state fixtures | The sport-specific state and transition interface can execute | Any real NBA accuracy, calibration, latency distribution on real data, or alpha |
| MLB | `BLOCKED_OR_INVENTORY_ONLY`: one Retrosheet real-game reducer replay plus inventory | The parsed event-to-state reducer can execute on the retained sample | A trained MLB probability model or market comparison |
| F1 | `BLOCKED_OR_INVENTORY_ONLY`: source/schema/license inventory | The candidate source fields and rights risks are documented | A reducer, trained model, calibrated probability, real-data latency, or alpha |

## Ranked reproduction decisions

### 1. NFL official fastrmodels feature specification

Decision: `PROPOSED_REPRODUCTION`, first priority.

The official nflverse training source defines a spread-aware win-probability
state containing second-half receiving team, spread adjusted for possession,
home/away possession, half and game seconds, score/time interaction, score
differential, down, yards to go, field position, and both teams' timeouts. The
official model is an XGBoost binary logistic model with monotonic constraints.

Primary sources:

- official model training source:
  <https://github.com/nflverse/fastrmodels/blob/master/data-raw/MODELS.R>
- official repository and MIT code license:
  <https://github.com/nflverse/fastrmodels> and
  <https://github.com/nflverse/fastrmodels/blob/master/LICENSE.md>
- public nflverse data releases:
  <https://github.com/nflverse/nflverse-data/releases/tag/pbp>

SAF must retrain this specification inside its frozen expanding-time folds. It
must not use the released `home_wp`, `wp`, `vegas_wp`, EPA, WPA, final score, or
postgame drive fields as features because the shipped artifact may contain
test-era information. The shortest comparison is:

- `official_no_spread_feature_spec`;
- `official_spread_feature_spec`;
- the existing SAF logistic and GBDT heads;
- the frozen spread-derived prior.

All models must use identical game/cutoff rows. This is a reproduction of a
mature public definition, not a claim that its published weights are valid
out-of-sample for SAF.

### 2. Soccer state-conditioned Cox/Poisson intensity

Decision: `PROPOSED_REPRODUCTION`, second priority.

Maia et al. define goal-event intensities using adapted regressors: team attack
and defence, home advantage, half/time, current score difference, and red-card
difference. "Adapted" means a value at time `t` depends only on events observed
by `t`. One fitted process can produce both:

- `home_goal | away_goal | no_goal` over the next 300 seconds; and
- final `home_win | draw | away_win` probabilities from the current state.

Primary source: <https://arxiv.org/abs/2312.04338>.

This is the shortest public-model reproduction that directly tests the current
SAF gap: the existing five-minute head repeats a pregame rate and does not use
the rich reducer state. The first version must use only score, clock/half, red
cards, and team attack/defence parameters known at the cutoff. Possession,
location, xG, and learned embeddings are excluded from v1 to avoid unregistered
feature search.

The paper reports that dynamic regressors improve in-game forecasting and gives
coefficient interpretations. Those are `PAPER_REPORTED`, not SAF results.
Because the paper's underlying Flashscore/CBF/Transfermarkt dataset and fitting
code are not released under a reusable data/code license, SAF may reproduce
only the published model family on the separately governed StatsBomb snapshot.

### 3. NFL next-drive full-state multinomial logistic

Decision: `PROPOSED_REPRODUCTION`, after the final-outcome comparator.

Keep the existing five target classes exactly:

`touchdown | field_goal | punt | turnover | other`.

Add only fields already emitted by the reducer and available at drive start:
down, yards to go, field position, goal-to-go, half/game clock, score
differential, possession, and both timeout counts. The existing next-drive head
uses the final-outcome feature subset and therefore omits the most direct drive
state. This change requires a preregistration amendment and a new model-registry
version before any fitted result.

### 4. NBA

Decision: `BLOCKED_OR_INVENTORY_ONLY`.

Yeh, Rice, and Dubin use normalized time, score differential, and an ESPN
pregame strength estimate in a pointwise logistic/probit model; Cervone et al.
define a tracking-data expected-possession-value state and next
macro-transition hazards:

- <https://arxiv.org/abs/2010.00781>
- <https://arxiv.org/abs/1408.0777>

Their results are `PAPER_REPORTED`. They do not provide a licensed complete
input dataset suitable for this program. NBA Terms also restrict gambling uses
and comprehensive live or archived play-by-play products:
<https://www.nba.com/termsofuse>. No real NBA model run is authorized until
Team I records written rights for the exact feed and research purpose.

### 5. MLB and F1

Decision: `BLOCKED_OR_INVENTORY_ONLY` for the current phase.

The mature MLB starting point is a 24 base/out-state run-expectancy Markov
model, using Retrosheet/Chadwick inputs. The mature public F1 references are the
TUM race simulator and Virtual Strategy Engineer:

- <https://www.retrosheet.org/game.htm>
- <https://chadwick.readthedocs.io/>
- <https://github.com/TUMFTM/race-simulation>
- <https://github.com/TUMFTM/f1-timing-database>

The Charter authorizes only inventory/evidence work for MLB/F1 in this phase.
Published F1 pit-model F1 scores or strategy results are not SAF calibration
evidence. FastF1's software license also does not itself grant rights to the
underlying Formula 1 timing data.

## Frozen evaluation and calibration protocol

Every future reproduction must pass all of the following before it can be
interpreted:

1. Split on calendar time and complete game. Every training game ends before
   the held-out game starts.
2. Freeze a separate earlier calibration interval. The base model, calibrator,
   and final test interval must be disjoint.
3. For repeated within-game predictions, compare identical fixed cutoffs and
   give games equal weight. Event-dense matches must not dominate.
4. Use a feature allowlist with an explicit as-of timestamp. Any future event,
   revised postgame field, released global WP/EPA/WPA, or final outcome fails
   closed.
5. Binary heads report Brier, log loss, calibration slope/intercept, reliability
   bins, and paired model-minus-prior differences.
6. Multiclass heads report multiclass Brier, log loss, classwise Brier,
   one-vs-rest slope/intercept, and reliability; soccer 1X2 also reports ranked
   probability score.
7. Fit sigmoid calibration for binary probabilities and a single frozen
   temperature for multiclass probabilities unless a different method was
   preregistered. The calibration interval may not be randomly mixed with the
   test interval.
8. Bootstrap complete games with a fixed seed and report requested and valid
   resamples. These intervals are descriptive because within-game states share
   one terminal result and even game-cluster bootstrap may under-cover.
9. Report model and full-path p50/p95/p99 latency on batch size one separately
   from training time, source-publication delay, and test-suite duration.
10. "Alpha" remains unobserved until the exact model cutoff is matched as-of to
    executable prediction-market bid/ask, with contract identity, resolution
    semantics, and latency preserved.

The scikit-learn calibration documentation explicitly requires independent
model-fitting and calibration data:
<https://scikit-learn.org/stable/modules/calibration.html>. Default unordered
cross-validation is therefore not accepted for these chronological streams.

## Required governance action before the next fitted run

No new empirical result may be produced from this review alone. Team H must
first append an experiment amendment that freezes:

- the new model ID/version;
- exact feature allowlist and cutoff semantics;
- source and training manifest hashes;
- parameter configuration and seed;
- chronological train/calibration/test folds;
- target/horizon and tie/terminal rules;
- bootstrap and calibration parameters.

Until that amendment exists, the official NFL and dynamic-soccer candidates are
research decisions, not registered results.
