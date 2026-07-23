# Program Status v0

- **Owner:** Team A
- **Version:** v0
- **Due gate:** `first_round_artifact_gate`
- **Decision:** `CONDITIONAL_GO`
- **Execution state:** `NO_REAL_MONEY`

The repository foundation, Team A contracts, Team H registry, immutable raw store, Team E label specification, Team F PRELIMINARY taker/TCA path, Team G audit workflow and reports, and Team I fail-closed matrices exist. Current checked-in evidence also includes the X-01 24-hour PMXT input preflight, the exact four-day X-02 input bundle, a bounded-incomplete Kalshi public REST prefix, the Polymarket recorder implementation and its current short observation, real-data X-11/X-12 POCs, and MLB/F1 source inventories. The X-09 fixture harness passes replay Levels 1+2. This is an implementation status statement, not a promotion decision or proof that a registered experiment is complete.

## Experiment state

- X-01: `DATA_READY_PREFLIGHT_ONLY`; the selected 2026-05-28 UTC day has a frozen 24-hour, 24-object input manifest and preflight. The preflight explicitly records `reconstruction_executed: false`; all-contract full-day semantic reconstruction, independent price/size comparison, and remaining registration locks are open.
- X-02: `DATA_READY_NO_RESULT`; the frozen bundle covers the four selected UTC days and 96 objects. Its bundle hash is registered, but the experiment card's `results_ref` remains empty and no timestamp measurement result or promotion decision is claimed.
- X-03: registered; one NBA market mapping exists, but the required four-week sport census is not complete.
- X-04: registered; no formal event-study execution has started because X-01 and X-02 are incomplete.
- X-05: specification complete; label generation remains blocked by unresolved locks and X-01.
- X-06: model/validation POC exists; empirical run is blocked by O-005 and unresolved data choices.
- X-07: pipeline output is PRELIMINARY only; formal and go/no-go scopes remain blocked.
- X-08: the recorder implementation and one 15-second, five-frame real Polymarket operational observation are verified. The observation is not a formal X-08 result and the seven-day gate is not met; persistent operation still needs a durable host. Kalshi public REST evidence is `BOUNDED_INCOMPLETE_LICENSE_PENDING`: markets and trades each verify 100 pages and 100,000 unique records, but both terminal cursors are nonempty, O-003 is pending, historical L2 is not claimed, and live/historical overlap was not run.
- X-09: deterministic fixture status is `HARNESS_PASS`; formal experiment state is `EXPERIMENT_BLOCKED` until its locks and X-01 clear.
- X-10: workflow/reports complete; experiment state is `REGISTERED_NOT_RUN` because the 50-pair sample and review protocol are not locked.
- X-11: `PRELIMINARY_POC`; the real-data full run and five governed model bindings exist with seed 20260722. Its method status remains `PIT_UNPROVEN`, the models remain `poc_only`, and the evidence is not eligible for a formal result.
- X-12: `POC_ONLY`; the real StatsBomb POC and two governed model bindings exist with seed 20260722. O-004 remains research-only, event availability is offline rather than live PIT, no point-in-time market prior exists, and formal promotion remains unauthorized.

The MLB and F1 deliverables are source inventories only. They do not claim a trained model, calibration result, or trading conclusion.

## Binding NO-GO

Real-money trading, live maker, multi-venue simultaneous live arbitrage, live copy trading, LLM hot path, RL, large-scale microservices, unregistered quick backtests, and README-return evidence remain prohibited. PMXT L2 is used only for deterministic level-2 book reconstruction; it cannot establish queue position or exact queue fill.

## User resources required

The next external inputs are: Kalshi live-L2 API key ID and RSA private-key path; intended operating jurisdiction/entity/account context; a persistent recorder host with durable storage and monitoring; WORM/object-lock policy for production raw data; an approved independent X-01 price/size comparison source; durable storage and retention for the X-03 four-week sport census; licensed point-in-time sports data or written permissions; and an approved external secret manager inventory/rotation policy with an incident owner. Team H must separately resolve each experiment's remaining registered locks before formal empirical acceptance.
