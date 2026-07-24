# NFL 2025 DAL@DET Game-State Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one byte-verified, deterministic offline game-state trace for the NFL game `2025_14_DAL_DET`, without reading or aligning any prediction-market data.

**Architecture:** A focused replay module verifies the governed frozen NFLVerse 2025 Parquet object, selects and orders the native game rows, and feeds the existing census-audited `NFLPlayEvent` adapter into the NFL reducer. It writes a content-addressed JSONL step trace and a compact evidence summary; source `time_of_day` is preserved as historical source time, never re-labelled as local receive/PIT time.

**Tech Stack:** Python 3.12, pyarrow Parquet, existing static-store verification, audited NFL game-state adapter/reducer, pytest.

---

## File structure

- Create: `src/prediction_market/sports/nfl_game_replay.py` — verified, single-native-game offline replay and deterministic artifact serialization.
- Create: `tests/sports/test_nfl_game_replay.py` — real frozen `2025_14_DAL_DET` proof of ordering, source-time accounting, deterministic hashes, and declared evidence boundary.
- Create at execution time: `artifacts/game-state/nfl/nfl_2025_14_dal_det_state_trace_v0.jsonl` — one immutable derived trace row per reducer transition.
- Create at execution time: `artifacts/game-state/nfl/nfl_2025_14_dal_det_state_replay_v0.json` — compact derived evidence report pointing at the immutable raw object and manifest.

### Task 1: Define the externally visible replay proof

**Files:**
- Create: `tests/sports/test_nfl_game_replay.py`
- Create: `src/prediction_market/sports/nfl_game_replay.py`

- [x] **Step 1: Write the failing test**

```python
from prediction_market.sports.nfl_game_replay import replay_frozen_nfl_game


def test_dal_det_replay_is_ordered_deterministic_and_offline_only() -> None:
    first = replay_frozen_nfl_game(
        program_root=PROJECT_ROOT,
        native_game_id="2025_14_DAL_DET",
    )
    second = replay_frozen_nfl_game(
        program_root=PROJECT_ROOT,
        native_game_id="2025_14_DAL_DET",
    )
    assert first.trace_sha256 == second.trace_sha256
    assert first.transition_count > 100
    assert first.native_order_strictly_increasing is True
    assert first.observation_mode == "offline_reconstruction"
    assert first.source_time_present_count == 187
    assert first.source_time_missing_count == 5
```

- [x] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/sports/test_nfl_game_replay.py -q`

Expected: FAIL during collection because `prediction_market.sports.nfl_game_replay` does not exist.

- [x] **Step 3: Write the minimal implementation**

```python
def replay_frozen_nfl_game(*, program_root: str | Path, native_game_id: str) -> NFLGameReplay:
    verified = read_verified_static_object(_FROZEN_MANIFEST_RELATIVE_PATH, ...)
    rows = _load_game_rows(verified.object_bytes, native_game_id)
    return _replay_rows(rows=rows, native_game_id=native_game_id, raw_sha256=...)
```

The implementation must:

1. use `read_verified_static_object`, pinned to the governed `I-018` season-2025 manifest;
2. use only the columns needed by `nflverse_transition_payload`, adding `time_of_day` only for timestamp accounting;
3. sort by integer `order_sequence` and reject duplicates or decreasing sequence;
4. derive each `NFLPlayEvent` ID from the immutable raw object hash, source version, ordered source-window IDs, native order sequences, and raw record ordinals through the existing census-audited adapter;
5. preserve each row’s `time_of_day` in the **derived trace only** with `source_time_status` of `present` or `missing_administrative`;
6. call `advance_state(nfl.NFLGameStateReducer(), state, event)` for every valid source row and record hash-chain material;
7. return an immutable report declaring `offline_reconstruction` and excluding market, PIT, model, alpha, and execution evidence.

- [x] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/sports/test_nfl_game_replay.py -q`

Expected: PASS.

### Task 2: Make the replay artifacts reproducible and scoped

**Files:**
- Modify: `tests/sports/test_nfl_game_replay.py`
- Modify: `src/prediction_market/sports/nfl_game_replay.py`

- [x] **Step 1: Write the failing artifact test**

```python
def test_written_artifacts_match_the_replayed_hash_chain(tmp_path: Path) -> None:
    report = write_frozen_nfl_game_replay(
        program_root=PROJECT_ROOT,
        native_game_id="2025_14_DAL_DET",
        output_directory=tmp_path,
    )
    summary = json.loads((tmp_path / report.summary_filename).read_text())
    trace_lines = (tmp_path / report.trace_filename).read_text().splitlines()
    assert summary["trace_sha256"] == report.trace_sha256
    assert len(trace_lines) == report.transition_count
    assert summary["evidence_boundary"]["market_data_included"] is False
    assert summary["evidence_boundary"]["pit_validated"] is False
```

- [x] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/sports/test_nfl_game_replay.py::test_written_artifacts_match_the_replayed_hash_chain -q`

Expected: FAIL because `write_frozen_nfl_game_replay` does not exist.

- [x] **Step 3: Write the minimal artifact writer**

```python
def write_frozen_nfl_game_replay(
    *, program_root: str | Path, native_game_id: str, output_directory: str | Path
) -> NFLGameReplay:
    report = replay_frozen_nfl_game(...)
    _write_jsonl_atomic(Path(output_directory) / report.trace_filename, report.steps)
    _write_json_atomic(Path(output_directory) / report.summary_filename, report.summary())
    return report
```

The JSONL entries must contain source order/play identity, `source_time_utc` or `null`, source-time status, event ID, previous/event/next state hashes, trace hash, and the next state. The summary must contain exact raw object and manifest hashes, reducer identity/version, game IDs/teams, counts, final-state hash, two-run trace hash equality, and this fixed boundary:

```json
{
  "market_data_included": false,
  "market_alignment_performed": false,
  "pit_validated": false,
  "model_output_included": false,
  "alpha_or_execution_claim": false
}
```

No output may be written under `var/raw`.

- [x] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/sports/test_nfl_game_replay.py -q`

Expected: PASS.

### Task 3: Generate the checked-in Phase-1 evidence

**Files:**
- Create: `artifacts/game-state/nfl/nfl_2025_14_dal_det_state_trace_v0.jsonl`
- Create: `artifacts/game-state/nfl/nfl_2025_14_dal_det_state_replay_v0.json`

- [x] **Step 1: Write the failing integration assertion**

```python
def test_checked_in_dal_det_evidence_is_current() -> None:
    generated = replay_frozen_nfl_game(program_root=PROJECT_ROOT, native_game_id="2025_14_DAL_DET")
    summary = json.loads((PROJECT_ROOT / "artifacts/game-state/nfl/nfl_2025_14_dal_det_state_replay_v0.json").read_text())
    assert summary["trace_sha256"] == generated.trace_sha256
```

- [x] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/sports/test_nfl_game_replay.py::test_checked_in_dal_det_evidence_is_current -q`

Expected: FAIL because the checked-in evidence does not exist.

- [x] **Step 3: Generate evidence with the verified writer**

Run:

```bash
PYTHONPATH=src ../../.venv/bin/python - <<'PY'
from pathlib import Path
from prediction_market.sports.nfl_game_replay import write_frozen_nfl_game_replay

worktree = Path.cwd()
write_frozen_nfl_game_replay(
    program_root=worktree.parent.parent,
    native_game_id="2025_14_DAL_DET",
    output_directory=worktree / "artifacts" / "game-state" / "nfl",
)
PY
```

- [x] **Step 4: Run the focused and related tests**

Run:

```bash
PYTHONPATH=src ../../.venv/bin/python -m pytest \
  tests/sports/test_nfl_game_replay.py \
  tests/sports/test_nfl_real_replay.py \
  tests/sports/test_nfl_season_state_artifact.py -q
```

Expected: PASS with no market-integration test invoked.

- [x] **Step 5: Review exact scope**

Run: `git status --short && git diff --check`

Expected: only the replay module, its tests, this plan, and the two derived evidence files are changed. No prediction-market fetcher, model, or raw object is modified.
