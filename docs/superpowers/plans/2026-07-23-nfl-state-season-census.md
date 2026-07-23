# NFL State-Transition Season Census Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-game NFL reducer-v2 proof with a reducer-v3 contract and a deterministic, field-audited census over all 285 frozen 2025 NFL games.

**Architecture:** Keep the shared `GameState` protocol structural and keep every football rule in the NFL module. Cleanly migrate the NFL state/event schema to include source ordering, season type, and suspension lifecycle; normalize postseason overtime timeout counters from the pinned 2025 NFL Rule 16 rather than accepting nflverse's negative counter. A separate census module must derive expected field values directly from frozen native rows and the pinned rulebook without calling the reducer's row adapter, so the reducer does not generate its own oracle.

**Tech Stack:** Python 3.13, frozen dataclasses, Pydantic `EventEnvelopeV0`, PyArrow, pytest, canonical SHA-256, nflverse Parquet release `58152862`, 2025 NFL Rulebook.

**Evidence boundary:** This is offline state-reconstruction engineering evidence. It is not a live PIT feed, a fitted X-11 result, model accuracy, prediction-market alignment, or alpha.

---

## Frozen decisions

- `order_sequence`, not Parquet row position or `play_id`, is the canonical within-game source order. Duplicate/non-increasing `order_sequence` fails closed.
- Regular-season overtime has 600-second periods and two timeouts per team.
- Postseason overtime has 900-second periods and three timeouts per team per overtime half. Periods 5–6 form the first overtime half; periods 7–8 form the next.
- For postseason rows in periods 5–6, nflverse timeout counters are normalized by the rules-derived offset required to start the overtime half at three. A negative normalized value still fails closed.
- A same-period clock increase is accepted only on an inserted timeout row ordered by `order_sequence`, with the dedicated quality flag and a non-monotone native `play_id` boundary. General clock increases remain invalid.
- `COMMENT` rows whose descriptions are exactly the observed suspend/resume lifecycle messages update a `suspended` state flag and carry clock/football context. Other missing-clock comments fail closed.
- The reducer remains explicitly `offline` because adjacent rows are used to observe the next football context. No live-causal claim is permitted.

Primary sources:

- 2025 NFL Rulebook Rule 16: `https://operations.nfl.com/media/ntif5hxb/2025-nfl-rulebook-final.pdf`
- nflfastR frozen helper commit: `ead5e2f9641490f692d923c04835bd3b90275b4e`
- nflverse frozen dataset release: `https://github.com/nflverse/nflverse-data/releases/tag/pbp`

## File responsibilities

- `src/prediction_market/sports/nfl_game_state.py`: NFL-only state/event schema, row normalization, lifecycle and transition invariants.
- `src/prediction_market/models/nfl.py`: full post-state binding for the official-model feature seam.
- `src/prediction_market/sports/nfl_season_census.py`: frozen-object loading, independent field oracle, two-run census, latency, evidence serialization.
- `tests/sports/test_nfl_game_state.py`: reducer-v3 unit and adversarial transition tests.
- `tests/models/test_nfl_fastrmodels.py`: model seam rejects any v3 state/envelope mismatch.
- `tests/sports/test_nfl_season_census.py`: synthetic census TDD plus real frozen-2025 census.
- `artifacts/game-state/nfl_season_state_validation_v2.json`: generated immutable engineering evidence.
- `artifacts/game-state/state_transition_research_v2.md`: primary-source reuse decisions and explicit oracle limits across NBA/NFL/soccer/MLB/F1.
- `artifacts/game-state/current_validation_report_zh_v1.md`: current Chinese truth table.
- `registries/artifact_registry.csv` and `artifacts/architecture/week1_backlog_v0.csv`: owner/version/gate/status.

### Task 1: Clean NFL reducer-v3 schema migration

**Files:**
- Modify: `tests/sports/test_nfl_game_state.py`
- Modify: `tests/models/test_nfl_fastrmodels.py`
- Modify: `src/prediction_market/sports/nfl_game_state.py`
- Create: `src/prediction_market/sports/nfl_rules.py`
- Modify: `src/prediction_market/models/nfl.py`

- [ ] **Step 1: Write failing schema and continuity tests**

Add required fields to every NFL fixture and add tests equivalent to:

```python
def test_reducer_rejects_source_identity_or_order_discontinuity() -> None:
    state = _state(
        season_type="REG",
        source_play_id="100",
        source_order_sequence=100,
        suspended=False,
    )
    event = _event(
        source_play_id="101",
        source_order_sequence=101,
        next_source_order_sequence=102,
    )
    with pytest.raises(NFLGameStateError, match="source"):
        reduce(state, event)


def test_postseason_overtime_uses_three_timeouts_per_half() -> None:
    state = _state(
        season_type="POST",
        period=4,
        period_seconds_remaining=0,
        game_seconds_remaining=0,
        home_timeouts_remaining=1,
        away_timeouts_remaining=0,
    )
    overtime = _event(
        season_type="POST",
        period=5,
        period_seconds_remaining=900,
        game_seconds_remaining=900,
        home_timeouts_remaining=3,
        away_timeouts_remaining=3,
        period_changed=True,
    )
    assert reduce(state, overtime).away_timeouts_remaining == 3


def test_regular_season_overtime_still_uses_two_timeouts() -> None:
    state = _regulation_end_tied_state(season_type="REG")
    overtime = _overtime_start_event(
        season_type="REG",
        seconds=600,
        home_timeouts_remaining=2,
        away_timeouts_remaining=2,
    )
    assert reduce(state, overtime).period_seconds_remaining == 600
```

Also add negative tests for:

- postseason period 5 starting at two;
- regular-season period 5 starting at three;
- postseason period 5→6 resetting timeouts;
- postseason period 5→6 resetting to three;
- arbitrary same-period clock increase;
- clock increase with a quality flag but without an inserted timeout boundary;
- arbitrary missing-clock comment;
- resume without a preceding suspension;
- second suspension while already suspended;
- `state.season_type != event.season_type`;
- a model feature context whose envelope/state source order, season type, or suspension flag differs.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```shell
.venv/bin/pytest -q \
  tests/sports/test_nfl_game_state.py \
  tests/models/test_nfl_fastrmodels.py
```

Expected: failures show that the new required fields, lifecycle rules, source-order continuity, and overtime rules do not yet exist.

- [ ] **Step 3: Implement the v3 schema without a compatibility layer**

Make these fields mandatory:

```python
@dataclass(frozen=True, slots=True)
class NFLGameState:
    sport: str
    game_id: str
    sequence: int
    terminal: bool
    season_type: str
    home_team: str
    away_team: str
    period: int
    period_seconds_remaining: int
    game_seconds_remaining: int
    source_play_id: str
    source_order_sequence: int
    suspended: bool
    # existing NFL-specific fields follow unchanged


@dataclass(frozen=True, slots=True)
class NFLPlayEvent:
    sport: str
    game_id: str
    sequence: int
    event_id: str
    season_type: str
    source_play_id: str
    source_order_sequence: int
    next_source_play_id: str
    next_source_order_sequence: int
    lifecycle_action: Literal["none", "suspend", "resume"]
    timeout_kind: Literal["none", "administrative", "play_attached"]
    clock_correction: bool
    clock_carry_forward: bool
    period: int | None
    period_seconds_remaining: int | None
    game_seconds_remaining: int | None
    # existing NFL-specific fields follow
```

Implement the following invariants in `reduce` and its validators:

```python
if event.source_play_id != state.source_play_id:
    raise NFLGameStateError("event source_play_id must match current state")
if event.source_order_sequence != state.source_order_sequence:
    raise NFLGameStateError(
        "event source_order_sequence must match current state"
    )
if event.next_source_order_sequence <= event.source_order_sequence:
    raise NFLGameStateError("next source order must strictly increase")
if event.season_type != state.season_type:
    raise NFLGameStateError("season_type cannot change within a game")
```

Use `clock_carry_forward=True` only for `suspend` and `resume`; those events must have all three clock fields `None`. Every other event must contain a fully validated clock. The reducer copies the prior clock for lifecycle events.

`timeout_kind="administrative"` may carry the observed football context while
charging exactly one timeout. `timeout_kind="play_attached"` must apply the
real play's observed context before charging the timeout; it may not freeze the
play. `timeout_kind="none"` must not consume a timeout. Keep the existing
`timeout`/`timeout_team` fields as the charged-counter assertion and require
them to agree with `timeout_kind`.

At lifecycle reduction:

```python
if event.lifecycle_action == "suspend":
    if state.suspended:
        raise NFLGameStateError("game is already suspended")
    target_suspended = True
elif event.lifecycle_action == "resume":
    if not state.suspended:
        raise NFLGameStateError("game is not suspended")
    target_suspended = False
else:
    target_suspended = state.suspended
```

Derive lifecycle only from `play_type_nfl == "COMMENT"` and the bounded source descriptions beginning with `The game has been suspended.` or `The game has resumed.`. Missing-clock rows outside those two lifecycle classes fail closed.

Normalize timeout counters with one rules function:

```python
def _timeout_allotment(season_type: str, period: int) -> int:
    if period <= 4:
        return 3
    if season_type == "REG":
        return 2
    return 3


def _postseason_ot_timeout_offset(
    season_type: str,
    period: int,
) -> int:
    return 1 if season_type == "POST" and period >= 5 else 0
```

Apply the offset to native postseason OT counters before validation. Reject normalized values outside `[0, 3]`. Timeout reset is allowed at halftime, at REG period 5, and at odd POST overtime periods 5/7/9; it is forbidden at POST periods 6/8.

Choose source clocks as follows:

- lifecycle current row: carry prior clock;
- inserted timeout current row: use the timeout row clock;
- when the next row is an inserted timeout: retain the current row clock until that timeout is consumed;
- otherwise: use the next observation clock as the existing offline adapter does.

Allow a same-period clock increase only when the current event is a timeout, `source_order_sequence` is strictly ordered, native `play_id` is inserted after the following play ID, and `quality_flags` contains `source_order_inserted_timeout`. Do not accept a caller-supplied flag without the native ordering predicate.

Bump:

```python
class NFLGameStateReducer:
    reducer_version = "v3"
```

Update `_validate_post_state_projection` in `models/nfl.py` to bind `season_type`, `source_order_sequence`, and `suspended` in addition to every existing state field.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```shell
.venv/bin/pytest -q \
  tests/sports/test_nfl_game_state.py \
  tests/models/test_nfl_fastrmodels.py \
  tests/sports/test_nfl_real_replay.py \
  tests/sports/test_model_latency.py
```

Expected: all focused tests pass; real tests may skip only when frozen raw objects are absent.

- [ ] **Step 5: Commit Task 1**

```shell
git add \
  src/prediction_market/sports/nfl_game_state.py \
  src/prediction_market/models/nfl.py \
  tests/sports/test_nfl_game_state.py \
  tests/models/test_nfl_fastrmodels.py \
  tests/sports/test_nfl_real_replay.py \
  tests/sports/test_model_latency.py
git commit -m "feat: model complete NFL source lifecycle state"
```

### Task 2: Build the independent 2025 season census

**Files:**
- Create: `tests/sports/test_nfl_season_census.py`
- Create: `src/prediction_market/sports/nfl_season_census.py`

- [ ] **Step 1: Write failing synthetic oracle tests**

The tests must construct small native-row games that cover:

- normal play;
- touchdown followed by PAT;
- timeout inserted by `order_sequence`;
- suspension then resume with missing clocks;
- same-period clock correction on an inserted timeout;
- REG OT timeout reset to two;
- POST OT timeout reset to three and third timeout reaching zero;
- duplicate/decreasing `order_sequence`;
- a native score mutation caught by the oracle;
- a context-carry mutation caught by the oracle;
- deterministic two-run output.

Require this public shape:

```python
@dataclass(frozen=True, slots=True)
class NFLFieldAudit:
    field: str
    basis: Literal[
        "native_direct",
        "native_derived",
        "rule_derived",
        "lineage",
    ]
    comparisons: int
    matches: int
    mismatches: int


@dataclass(frozen=True, slots=True)
class NFLSeasonCensusReport:
    census_version: str
    reducer_version: str
    dataset_id: str
    source_version: str
    rulebook_version: str
    scan_runs: int
    deterministic: bool
    games_total: int
    completed_games: int
    fail_closed_games: int
    transitions: int
    field_audits: tuple[NFLFieldAudit, ...]
    lifecycle_counts: tuple[tuple[str, int], ...]
    quality_flag_counts: tuple[tuple[str, int], ...]
    canonical_state_sha256: str
    reducer_latency: Mapping[str, int]
```

The independent oracle module must not call:

- `state_from_nflverse_row`;
- `_snapshot_from_row`;
- `nflverse_transition_payload`;
- `reduce` to compute expected values.

It may call the reducer only to obtain the actual state being evaluated.

- [ ] **Step 2: Run synthetic census tests and verify RED**

Run:

```shell
.venv/bin/pytest -q tests/sports/test_nfl_season_census.py
```

Expected: import failure because `nfl_season_census.py` does not exist.

- [ ] **Step 3: Implement native loading, independent field expectations, and two-run hashing**

Implement:

```python
def census_loaded_nflverse_season(
    *,
    rows: Sequence[Mapping[str, object]],
    raw_object_sha256: str,
    source_version: str,
    scan_runs: int = 2,
) -> NFLSeasonCensusReport:
    ...


def run_frozen_nflverse_2025_census(
    *,
    program_root: str | Path,
) -> NFLSeasonCensusReport:
    ...
```

`run_frozen_nflverse_2025_census` must:

1. locate the registered 2025 manifest;
2. use `read_verified_static_object`;
3. recompute and require the object SHA-256;
4. load only the frozen required columns;
5. retain raw record ordinals;
6. group by `game_id`;
7. sort each group by `order_sequence`;
8. reject duplicate or non-integer order values;
9. execute two complete scans.

For every actual state, independently compare these categories:

- `native_direct`: game/team identity, season type, source play/order, period, available clocks, terminal marker;
- `native_derived`: mapped post-play score, possession, down/distance/spot, drive, play clock, timeout team/counter;
- `rule_derived`: suspension lifecycle, missing-clock carry, POST OT timeout offset/reset, inserted-timeout clock correction;
- `lineage`: sequence and last event ID.

The oracle must explicitly classify context-carry rows; it must never fill absent context from a future row. Report a mismatch instead of mutating expected values to match the reducer.

Time only the reducer call using `time.perf_counter_ns`. Report samples, p50, p95, p99, max, mean, and operations/second. Exclude Parquet, envelope material, row normalization, oracle construction, and model inference.

- [ ] **Step 4: Run synthetic tests and verify GREEN**

Run:

```shell
.venv/bin/pytest -q tests/sports/test_nfl_season_census.py -k "not frozen"
```

Expected: all synthetic census tests pass.

- [ ] **Step 5: Run the real frozen-2025 census**

Run:

```shell
.venv/bin/pytest -q \
  tests/sports/test_nfl_season_census.py \
  --basetemp=/private/var/folders/s9/hy_y_ljx629gf7t_7gs2wy0c0000gn/T/saf-nfl-census-pytest
```

Acceptance:

- `games_total == 285`;
- `completed_games == 285`;
- `fail_closed_games == 0`;
- two complete scans have the same canonical hash;
- every audited field has zero mismatches;
- two suspend and two resume lifecycle rows are present;
- exactly the actual source clock corrections are counted rather than allowing all clock increases;
- the postseason third-timeout row ends at zero, never `-1`.

If any acceptance condition is false, keep the artifact absent and report the exact game/order/field mismatch.

- [ ] **Step 6: Commit Task 2**

```shell
git add \
  src/prediction_market/sports/nfl_season_census.py \
  tests/sports/test_nfl_season_census.py
git commit -m "feat: validate NFL state transitions across 2025"
```

### Task 3: Publish evidence and research decisions

**Files:**
- Create: `artifacts/game-state/nfl_season_state_validation_v2.json`
- Create: `artifacts/game-state/state_transition_research_v2.md`
- Modify: `artifacts/game-state/current_validation_report_zh_v1.md`
- Modify: `registries/artifact_registry.csv`
- Modify: `artifacts/architecture/week1_backlog_v0.csv`
- Modify: `tests/test_experiment_registry.py`

- [ ] **Step 1: Add a failing artifact-registry test**

Require exactly one new row:

```text
ART-D2-003,artifacts/game-state/nfl_season_state_validation_v2.json,D2+C+H,v2,2026-08-05_W2_review,PRELIMINARY_ENGINEERING_VALIDATION
```

Require the artifact to identify `X-11` only as a registered reference and to set all of these false:

```json
{
  "formal_experiment_result": false,
  "model_evaluation": false,
  "prediction_market_alpha_evidence": false
}
```

Run:

```shell
.venv/bin/pytest -q tests/test_experiment_registry.py
```

Expected: failure until the artifact and registry row exist.

- [ ] **Step 2: Generate the immutable census artifact**

Add a CLI entry in `nfl_season_census.py` that writes canonical, sorted, indented JSON only after all Task 2 acceptance conditions pass. The artifact must include:

- dataset registry ID and record hash;
- source manifest path/hash and raw object hash;
- 2025 Rulebook URL and SHA-256 of a frozen local rule snapshot if one is stored;
- code/test hashes;
- all field-audit counts;
- lifecycle and quality-flag counts;
- two-run canonical hashes;
- reducer-only latency;
- explicit adjacent-row offline observation limitation;
- explicit statement that no fitted model or market data was used.

Run the generator twice and require byte-identical output before adding the file.

- [ ] **Step 3: Write the cross-sport primary-source decision record**

`state_transition_research_v2.md` must state:

- NFL: reuse nflverse ordering and fastrmodels; the new census is offline reconstruction, not a live predictor.
- Soccer: socceraction/Atomic-SPADL is a differential representation, not an independent oracle because it consumes the same StatsBomb events and discards lifecycle events; prefer the OpenSTARLab event model only after a frozen UEID adapter and registration.
- NBA: remain blocked until a licensed feed explicitly permits archived/live PBP and prediction-market research.
- MLB: next priority is full-2025 current-play reduction with separate next-row/game-log oracle, then the published 24-state Markov baseline.
- F1: reduced Jolpica research state is possible; TUMFTM is the reusable model candidate, but live/commercial rights remain blocked.

For calibration, freeze:

- complete-game chronological train/calibration/test partitions;
- game-equal calibration weighting;
- untouched final test;
- multiclass temperature only when all classes have calibration support;
- isotonic only when a preregistered calibration sample-size threshold is met;
- calibration slope/intercept, Brier, log loss, game-cluster bootstrap CI;
- no README metrics or pretrained predictions as SAF evidence.

- [ ] **Step 4: Update the Chinese report without overstating completion**

Replace the one-game NFL state statement only if the real census artifact exists and validates. Keep:

- NFL model metrics labelled as the separate historical X-11 POC;
- current governed model latency absent;
- `matched_as_of_rows=0`;
- alpha and symmetry unvalidated.

- [ ] **Step 5: Verify registries and commit Task 3**

Run:

```shell
.venv/bin/python tools/validate_governance.py
.venv/bin/pytest -q tests/test_experiment_registry.py
git diff --check
```

Then:

```shell
git add \
  artifacts/game-state/nfl_season_state_validation_v2.json \
  artifacts/game-state/state_transition_research_v2.md \
  artifacts/game-state/current_validation_report_zh_v1.md \
  registries/artifact_registry.csv \
  artifacts/architecture/week1_backlog_v0.csv \
  tests/test_experiment_registry.py
git commit -m "docs: publish NFL season state evidence"
```

### Task 4: Independent reviews and branch verification

**Files:**
- Review all files changed in Tasks 1–3.

- [ ] **Step 1: Run spec-compliance review**

The reviewer must prove:

- source ordering uses `order_sequence`;
- the four historical failure classes are handled by general rules, not game IDs;
- same-period clock monotonicity is still fail-closed outside the one explicit inserted-timeout class;
- POST OT timeout counts follow Rule 16 and never trust negative native values;
- lifecycle rows are represented in state;
- expected fields are built outside the reducer adapter;
- no model/alpha claim was introduced.

- [ ] **Step 2: Run code-quality review**

Reject:

- any compatibility alias for reducer v2;
- hard-coded game IDs;
- tests that only assert hashes without field values;
- oracle code that calls adapter helpers;
- a generated artifact not reproducible from frozen raw bytes;
- broad exception swallowing.

- [ ] **Step 3: Run the full suite in a bounded temp root**

Run:

```shell
.venv/bin/pytest -q \
  --basetemp=/private/var/folders/s9/hy_y_ljx629gf7t_7gs2wy0c0000gn/T/saf-nfl-v3-full-pytest
.venv/bin/python tools/validate_governance.py
git diff --check
```

Delete only that exact pytest temp directory after recording results.

- [ ] **Step 4: Finish the branch**

Use `superpowers:finishing-a-development-branch`. Fast-forward only after all reviews and tests pass; then push `main` to `https://github.com/WeKruit/SAF.git` and verify local/remote commit equality.
