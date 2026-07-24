"""Verified, offline replay evidence for one frozen NFLVerse game.

This module deliberately stops at a deterministic game-state trace.  It does
not fetch, read, align, or infer anything about prediction markets.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from prediction_market.contracts import canonical_json_bytes, canonical_sha256
from prediction_market.sports import nfl_game_state as nfl
from prediction_market.sports import nfl_season_census as census
from prediction_market.sports.game_state import (
    advance_state,
    canonical_state_sha256,
)
from prediction_market.sports.x11 import (
    X11_FROZEN_PARTITION_ALLOWLIST,
    X11_NFLVERSE_VERSION,
)
from prediction_market.static_store import (
    StaticStoreError,
    read_verified_static_object,
)


_ARTIFACT_VERSION = "v0"
_OBSERVATION_MODE: Literal["offline_reconstruction"] = "offline_reconstruction"
_DATASET_ID = "DS-NFLVERSE"
_EVIDENCE_BOUNDARY = {
    "market_data_included": False,
    "market_alignment_performed": False,
    "pit_validated": False,
    "model_output_included": False,
    "alpha_or_execution_claim": False,
}
_REPLAY_COLUMNS = (
    "game_id",
    "play_id",
    "order_sequence",
    "time_of_day",
    "season_type",
    "qtr",
    "quarter_seconds_remaining",
    "game_seconds_remaining",
    "home_team",
    "away_team",
    "fixed_drive",
    "goal_to_go",
    "play_clock",
    "posteam",
    "posteam_type",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_score",
    "defteam_score",
    "posteam_score_post",
    "defteam_score_post",
    "total_home_score",
    "total_away_score",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "play_type",
    "play_type_nfl",
    "desc",
    "sp",
    "first_down",
    "interception",
    "fumble_lost",
    "timeout",
    "timeout_team",
    "quarter_end",
    "series_result",
)


class NFLGameReplayError(ValueError):
    """A supplied frozen object cannot prove the requested offline trace."""


@dataclass(frozen=True, slots=True)
class NFLGameReplay:
    """One deterministic game-state replay and its bounded evidence claim."""

    artifact_version: str
    native_game_id: str
    canonical_game_id: str
    home_team: str
    away_team: str
    dataset_id: str
    source_version: str
    source_object_sha256: str
    source_manifest_sha256: str
    reducer_id: str
    reducer_version: str
    observation_mode: Literal["offline_reconstruction"]
    source_row_count: int
    transition_count: int
    source_time_present_count: int
    source_time_missing_count: int
    native_order_strictly_increasing: bool
    trace_sha256: str
    final_state_sha256: str
    steps: tuple[dict[str, Any], ...]
    evidence_boundary: dict[str, bool]

    @property
    def trace_filename(self) -> str:
        return f"nfl_{self.native_game_id.lower()}_state_trace_{self.artifact_version}.jsonl"

    @property
    def summary_filename(self) -> str:
        return f"nfl_{self.native_game_id.lower()}_state_replay_{self.artifact_version}.json"

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_type": "nfl_game_state_replay",
            "artifact_version": self.artifact_version,
            "native_game_id": self.native_game_id,
            "canonical_game_id": self.canonical_game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "dataset_id": self.dataset_id,
            "source_version": self.source_version,
            "source_object_sha256": self.source_object_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "reducer_id": self.reducer_id,
            "reducer_version": self.reducer_version,
            "observation_mode": self.observation_mode,
            "source_row_count": self.source_row_count,
            "transition_count": self.transition_count,
            "source_time": {
                "field": "nflverse.time_of_day",
                "present_row_count": self.source_time_present_count,
                "missing_row_count": self.source_time_missing_count,
                "semantics": "historical_source_event_timestamp_not_local_receive_time",
            },
            "native_order_strictly_increasing": self.native_order_strictly_increasing,
            "trace_sha256": self.trace_sha256,
            "final_state_sha256": self.final_state_sha256,
            "determinism": {
                "replay_runs": 2,
                "trace_hashes_equal": True,
            },
            "evidence_boundary": dict(self.evidence_boundary),
        }


def _required_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise NFLGameReplayError(f"{field} must be canonical nonempty text")
    return value


def _validated_order(row: Mapping[str, object]) -> int:
    value = row.get("order_sequence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NFLGameReplayError("order_sequence must be an integer")
    if int(value) != value or int(value) < 0:
        raise NFLGameReplayError("order_sequence must be a non-negative integer")
    return int(value)


def _source_time(row: Mapping[str, object]) -> tuple[str | None, str]:
    value = row.get("time_of_day")
    if value is None:
        return None, "missing_administrative"
    return _required_text(value, field="time_of_day"), "present"


def _load_verified_game_rows(
    *, program_root: Path, native_game_id: str
) -> tuple[tuple[dict[str, object], ...], str, str]:
    store_root = program_root / "var" / "raw"
    manifest_path = census._frozen_manifest_path(store_root)
    try:
        verified = read_verified_static_object(
            manifest_path,
            store_root=store_root,
            program_root=program_root,
        )
    except StaticStoreError as error:
        raise NFLGameReplayError(
            "frozen NFLVerse object failed static manifest verification"
        ) from error

    expected_object, _ = X11_FROZEN_PARTITION_ALLOWLIST[2025]
    object_sha256 = "sha256:" + hashlib.sha256(verified.object_bytes).hexdigest()
    if (
        verified.record.dataset != _DATASET_ID
        or verified.record.version != X11_NFLVERSE_VERSION
        or verified.record.partition != "season-2025"
        or verified.record.manifest.object_sha256 != expected_object
        or object_sha256 != expected_object
    ):
        raise NFLGameReplayError(
            "frozen NFLVerse source identity or content hash is not canonical"
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        rows = pq.read_table(
            pa.BufferReader(verified.object_bytes),
            columns=list(_REPLAY_COLUMNS),
        ).to_pylist()
    except (pa.ArrowException, OSError, ValueError) as error:
        raise NFLGameReplayError(
            "verified NFLVerse object is not readable as the governed Parquet schema"
        ) from error

    selected: list[dict[str, object]] = []
    for ordinal, native_row in enumerate(rows):
        row = dict(native_row)
        if row.get("game_id") != native_game_id:
            continue
        row["_raw_record_ordinal"] = ordinal
        selected.append(row)
    if len(selected) < 2:
        raise NFLGameReplayError(
            f"frozen NFLVerse object does not contain complete game {native_game_id}"
        )

    ordered = tuple(sorted(selected, key=_validated_order))
    orders = tuple(_validated_order(row) for row in ordered)
    if any(later <= earlier for earlier, later in zip(orders, orders[1:])):
        raise NFLGameReplayError(
            "native order_sequence must be strictly increasing and unique"
        )
    return (
        ordered,
        object_sha256,
        verified.record.manifest.manifest_sha256,
    )


def _replay_rows(
    *,
    rows: Sequence[dict[str, object]],
    native_game_id: str,
    source_object_sha256: str,
    source_manifest_sha256: str,
) -> NFLGameReplay:
    state = nfl.state_from_nflverse_row(rows[0], sequence=0)
    steps: list[dict[str, Any]] = []
    for sequence, (source_row, successor_row) in enumerate(
        zip(rows, rows[1:]), start=1
    ):
        successors = census._causal_successor_window(
            rows,
            source_row=source_row,
            post_index=sequence,
        )
        event = census._actual_event(
            state,
            source_row,
            successor_row,
            successor_rows=successors,
            sequence=sequence,
            raw_object_sha256=source_object_sha256,
            source_version=X11_NFLVERSE_VERSION,
        )
        trace = advance_state(nfl.NFL_GAME_STATE_REDUCER, state, event)
        source_time_utc, source_time_status = _source_time(source_row)
        steps.append(
            {
                "sequence": sequence,
                "source_play_id": event.source_play_id,
                "source_order_sequence": event.source_order_sequence,
                "source_time_utc": source_time_utc,
                "source_time_status": source_time_status,
                "event_id": event.event_id,
                "previous_state_sha256": trace.previous_state_sha256,
                "event_sha256": trace.event_sha256,
                "next_state_sha256": trace.next_state_sha256,
                "step_trace_sha256": trace.trace_sha256,
                "next_state": asdict(trace.next_state),
            }
        )
        state = trace.next_state

    all_source_times = tuple(_source_time(row) for row in rows)
    trace_sha256 = canonical_sha256(
        {
            "artifact_version": _ARTIFACT_VERSION,
            "native_game_id": native_game_id,
            "source_object_sha256": source_object_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "steps": steps,
        }
    )
    return NFLGameReplay(
        artifact_version=_ARTIFACT_VERSION,
        native_game_id=native_game_id,
        canonical_game_id=state.game_id,
        home_team=state.home_team,
        away_team=state.away_team,
        dataset_id=_DATASET_ID,
        source_version=X11_NFLVERSE_VERSION,
        source_object_sha256=source_object_sha256,
        source_manifest_sha256=source_manifest_sha256,
        reducer_id=nfl.NFL_GAME_STATE_REDUCER.reducer_id,
        reducer_version=nfl.NFL_GAME_STATE_REDUCER.reducer_version,
        observation_mode=_OBSERVATION_MODE,
        source_row_count=len(rows),
        transition_count=len(steps),
        source_time_present_count=sum(
            status == "present" for _, status in all_source_times
        ),
        source_time_missing_count=sum(
            status == "missing_administrative" for _, status in all_source_times
        ),
        native_order_strictly_increasing=True,
        trace_sha256=trace_sha256,
        final_state_sha256=canonical_state_sha256(state),
        steps=tuple(steps),
        evidence_boundary=dict(_EVIDENCE_BOUNDARY),
    )


def replay_frozen_nfl_game(
    *, program_root: str | Path, native_game_id: str
) -> NFLGameReplay:
    """Replay one native NFL game from the byte-verified frozen 2025 season."""

    root = Path(program_root).resolve()
    _required_text(native_game_id, field="native_game_id")
    rows, source_object_sha256, source_manifest_sha256 = _load_verified_game_rows(
        program_root=root,
        native_game_id=native_game_id,
    )
    first = _replay_rows(
        rows=rows,
        native_game_id=native_game_id,
        source_object_sha256=source_object_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )
    second = _replay_rows(
        rows=rows,
        native_game_id=native_game_id,
        source_object_sha256=source_object_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )
    if first.trace_sha256 != second.trace_sha256:
        raise NFLGameReplayError("two deterministic replay runs produced different hashes")
    return first


def _write_atomic(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_frozen_nfl_game_replay(
    *,
    program_root: str | Path,
    native_game_id: str,
    output_directory: str | Path,
) -> NFLGameReplay:
    """Write derived replay evidence outside the immutable raw-store root."""

    report = replay_frozen_nfl_game(
        program_root=program_root,
        native_game_id=native_game_id,
    )
    output_root = Path(output_directory).resolve()
    raw_root = Path(program_root).resolve() / "var" / "raw"
    if output_root == raw_root or raw_root in output_root.parents:
        raise NFLGameReplayError("derived replay output cannot be written under var/raw")
    _write_atomic(
        output_root / report.trace_filename,
        b"".join(canonical_json_bytes(step) + b"\n" for step in report.steps),
    )
    _write_atomic(
        output_root / report.summary_filename,
        canonical_json_bytes(report.summary()) + b"\n",
    )
    return report
