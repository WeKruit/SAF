# X-12 Real-Data Dixon-Coles POC v0

- **Owner:** Team D3 + C + H
- **Experiment:** `X-12`
- **Authorized scope:** `poc_result`
- **Result label:** `POC_NO_PIT_MARKET_PRIOR`
- **Contract result class:** `PRELIMINARY`
- **Formal-result eligible:** `false`
- **Source:** `DS-STATSBOMB-OPEN`, Premier League 2015/16, competition `2`,
  season `27`
- **Frozen commit:** `b0bc9f22dd77c206ddedc1d742893b3bbe64baec`
- **License binding:** `O-004`, `research_only`
- **Machine-readable evidence:** `x12_real_data_poc_v0.json`

## Frozen input and lineage

- 1 match-index manifest plus 380 event manifests were read from the main
  immutable raw store and verified before adaptation.
- Matches: 380
- Native events: 1,313,773
- Reconciled goals: 1,026
- Teams: 20
- Inventory SHA-256:
  `sha256:f45bca3335610821e1d49783d274c80c6b532be1c43002a1406393316f1a54cf`
- Chronology SHA-256:
  `sha256:a9912bb9a33ba9003d5b5e5457ba7a09d1af00e256cd663ec8cb02c6e8527869`
- Goal-timeline SHA-256:
  `sha256:42f9bb1ab04fb41fd7c7b60d4355d069d0fb605508275b4aca6220a70e09b178`
- Evidence self-hash:
  `sha256:c455191f2d619ce68ac2b2059a5e2739a7823b4940e8c21f1b2a554bbd7887b1`
- Evidence file SHA-256:
  `sha256:34b80e999885f3c44c0a9366b7595ed24080623e42fc2e4e4ed2c264fceb61eb`

The adapter observed 512 native event-index/time regressions. They are retained
and reported rather than silently treated as a live point-in-time sequence.
Goal labels use native `(minute, second, index)` ordering. This is therefore an
offline reconstruction POC, not evidence of live event availability.

## Evaluation

The evaluator used a date-grouped chronological expanding window. Each training
match is on a calendar date strictly before its test date, and the conservative
training outcome availability time (`kickoff + 3 hours`) is strictly before the
prediction cutoff. The first 100 matches form the minimum training window.
The Dixon-Coles likelihood is fitted with an analytic gradient and bounded
SLSQP. Every fold fails closed unless it either makes measurable objective and
parameter progress or begins at a verified stationary point, and every accepted
solution must pass an element-by-element initial and final parameter-bound check,
with only an explicit eight-ULP machine-roundoff allowance, and independently
remain inside the finite Dixon-Coles likelihood domain. KKT projection applies
only within the explicit `1e-10` boundary proximity tolerance and only after
the bound check, so a genuinely out-of-bounds result cannot be projected away.
The objective must not worsen and projected gradient infinity norm must be at
most `1e-4`. An invalid-tau sentinel can never qualify as a stationary solution.

- Held-out matches: 280
- Expanding folds: 72
- Distinct fitted parameter hashes: 72
- Distinct held-out expected-goal pairs: 280
- Minimum fold objective improvement: `0.005786870654731047`
- Minimum fold parameter displacement: `0.03164255012865628`
- Maximum accepted projected gradient infinity norm:
  `0.000049174873211848126`
- Match-cluster bootstrap: seed `20260722`, requested 200, valid 200
- Multiclass Brier: `0.6333086250422754`
- Multiclass log loss: `1.0566358383349692`
- One-vs-rest calibration:
  - home win: slope `0.7442644081974514`, intercept
    `0.02552324397646033`
  - draw: slope `-0.13257622504791863`, intercept
    `-1.08045784291297`
  - away win: slope `0.5609461157441388`, intercept
    `-0.5311338373234309`

The comparator is the strictly expanding empirical 1X2 frequency with
Laplace-1 smoothing; it is not represented as a market prior.

- Model-minus-baseline Brier delta: `-0.024442623053046564`
- Brier delta 95% clustered CI:
  `[-0.06037747745710088, 0.013990344957441411]`
- Model-minus-baseline log-loss delta: `-0.029168652881823798`
- Log-loss delta 95% clustered CI:
  `[-0.08353985226881108, 0.028878288366651978]`

Both intervals cross zero. This POC does not establish improvement over the
simple baseline.

## Five-minute transition output

The run emitted 5,040 home-goal / away-goal / no-goal distributions: 18
five-minute snapshots for each held-out match. Every distribution is converted
to exact fixed-point probabilities and structurally validated against
`model-output/v1`; a representative output also passed the registry-backed
normative validator.

- Transition Brier: `0.24552780282114847`
- Transition log loss: `0.49219732711024167`
- Match-cluster bootstrap: requested 200, valid 200
- Availability label: `offline_reconstruction_not_live_PIT`

## Open gates

- Team H registration-lock approval remains open.
- No point-in-time market prior exists for this source snapshot.
- StatsBomb `O-004` remains research-only.
- Offline archive timestamps do not prove live point-in-time availability.
- Formal promotion is unauthorized.

All program NO-GOs remain in force. This artifact is not a trading result, does
not authorize real-money action, and does not enter a production trading path.
