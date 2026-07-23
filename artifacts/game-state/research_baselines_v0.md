# Game-State Research and Reproduction Matrix v0

Date: 2026-07-23  
Owner: Team D + H + C + I  
Status: evidence review complete; reproduction in progress  
Decision class: research-only, no trading conclusion

## Validation research

The program selects probability models by calibration and proper scoring rules,
not classification accuracy:

- Brill, Yurko, and Wyner show that observations within one game are highly
  dependent and that treating plays as independent overstates effective sample
  size. SAF therefore splits and bootstraps by complete game:
  <https://arxiv.org/abs/2406.16171>.
- Walsh and Joshi argue that sports-betting model selection depends more on
  calibrated probabilities than raw accuracy. SAF reports Brier, log loss,
  calibration slope/intercept, and reliability:
  <https://arxiv.org/abs/2303.06021>.

Calibration is necessary but not sufficient for alpha. A calibrated model may
match the market, react later than the market, or disagree only inside the
spread.

## Reproduction matrix

| Sport | Mature work reviewed | State/target reproduced first | Data and rights boundary | Decision |
|---|---|---|---|---|
| NFL | Lock & Nettleton random-forest WP; nflWAR/nflfastR EP/WP; NFLSimulatoR state simulation | Play-level reducer with clock, score, possession, down/distance, field position and timeouts; next-drive outcome and final WP are separate heads | nflverse is CC-BY-4.0 and suitable for frozen offline work; its release data is not a live sports feed | Implement first; compare to nflfastR before optimizing |
| Soccer | Dixon-Coles; Robberechts Bayesian in-game WP; Maia dynamic Cox intensities; SPADL/xT/VAEP; NMSTPP next-event forecasting | Event reducer with score, clock, cards, substitutions, possession, location and active players; state-conditioned five-minute goal transition | StatsBomb Open Data is research-only under the current Team I decision and has no live PIT publication timestamp | Implement reducer; replace the current constant match-rate transition with an interpretable state-conditioned model |
| MLB | Retrosheet/Chadwick; Bukiet et al. 24 base/out Markov model | Parsed-play reducer for inning, outs, bases and score; next plate-appearance transition and half-inning run distribution | Retrosheet requires its own attribution/usage terms; Chadwick is the canonical parser and is not currently installed | Implement reducer against parsed observations; install/verify Chadwick before empirical claims |
| NBA | Song/Shi gamma process; bookmaker-prior gamma model; Vračar play-by-play model; Cervone EPV | Possession reducer and next-possession score transition after a licensed feed exists | `nba_api` code licensing does not grant NBA data rights; current NBA use is blocked by O-005 | Interface/fixtures only; do not use tracking EPV as the first reproducible baseline |
| F1 | FastF1/Jolpica; TUM race simulation; Virtual Strategy Engineer | Lap-level reducer with gaps, tyres/stints, pit, SC/VSC, weather and race-control state | FastF1 code does not grant underlying F1 data rights and does not prove a real-time SLA | Interface/fixtures only until Team I clears the data |

## Primary sources

### NFL

- Lock & Nettleton, *Using Random Forests to Estimate Win Probability Before
  Each Play of an NFL Game*:
  <https://doi.org/10.1515/jqas-2013-0100>
- Yurko, Ventura, and Horowitz, *nflWAR*:
  <https://arxiv.org/abs/1802.00998>
- nflverse data and nflfastR model description:
  <https://github.com/nflverse/nflverse-data> and
  <https://opensourcefootball.com/posts/2020-09-28-nflfastr-ep-wp-and-cp-models/>
- NFLSimulatoR:
  <https://arxiv.org/abs/2102.01846>

### Soccer

- Dixon & Coles:
  <https://doi.org/10.1111/1467-9876.00065>
- Robberechts et al.:
  <https://arxiv.org/abs/1906.05029>
- Maia et al.:
  <https://arxiv.org/abs/2312.04338>
- VAEP and socceraction:
  <https://arxiv.org/abs/1802.07127> and
  <https://github.com/ML-KULeuven/socceraction>
- NMSTPP:
  <https://doi.org/10.1007/s10489-024-05996-9>

### MLB

- Retrosheet event files and Chadwick:
  <https://www.retrosheet.org/game.htm> and
  <https://chadwick.readthedocs.io/en/stable/>
- Bukiet, Harold, and Palacios, 24-state Markov model:
  <https://doi.org/10.1287/opre.45.1.14>

### NBA

- Song & Shi gamma-process model:
  <https://www.sciencedirect.com/science/article/pii/S0377221719309233>
- Bookmaker-prior gamma model:
  <https://www.sciencedirect.com/science/article/pii/S0378437120301618>
- Vračar et al. play-by-play model:
  <https://www.sciencedirect.com/science/article/abs/pii/S095741741500617X>
- Cervone et al. EPV:
  <https://arxiv.org/abs/1408.0777>
- NBA terms:
  <https://www.nba.com/termsofuse>

### F1

- FastF1 data reference and live timing:
  <https://docs.fastf1.dev/data_reference/index.html> and
  <https://docs.fastf1.dev/livetiming.html>
- TUM race simulation:
  <https://github.com/TUMFTM/race-simulation>
- Virtual Strategy Engineer:
  <https://doi.org/10.3390/app10217805>

## Prediction-market evidence boundary

External research suggests that model-market differences must be studied as a
time-aligned response, not a final prediction comparison:

- Angelini and De Angelis study contemporaneous model-versus-market updating
  and subsequent drift: <https://arxiv.org/abs/2606.07811>.
- Clegg et al. use a market-calibrated football prior, but their simulated
  return claims still require independent reproduction:
  <https://arxiv.org/abs/2605.16066>.
- A Polymarket NBA study reports rare, short, depth-limited executable windows;
  this is external evidence, not a SAF result:
  <https://arxiv.org/abs/2605.00864>.

SAF will therefore use the neutral label `predictive_disagreement` until a
same-game, matched-as-of, executable bid/ask comparison passes its registered
gate.

## Latency finding

The reviewed work does not provide comparable batch-one reducer, feature,
inference, and end-to-end p50/p95/p99 measurements. SAF must measure all four
stages on the exact reproduced implementation. Training time, data publication
delay, and repository test duration are not inference latency.
