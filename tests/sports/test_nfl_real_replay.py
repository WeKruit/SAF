from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from prediction_market.contracts import EventEnvelopeV0
from prediction_market.experiments import load_experiment_registry
from prediction_market.sports.game_state import canonical_state_sha256
from prediction_market.sports import nfl_game_state as nfl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_OBJECT_SHA256 = (
    "sha256:3730c4db2ab99d2dfc4017de975b7610c46c35301b9280b65c03de1b1c74265a"
)
NFLVERSE_2025 = (
    PROJECT_ROOT
    / "var"
    / "raw"
    / "raw"
    / "source=nflverse"
    / "dataset=DS-NFLVERSE"
    / "version=github-release-58152862-20260212T102526Z"
    / "partition=season-2025"
    / f"{RAW_OBJECT_SHA256.removeprefix('sha256:')}.parquet"
)
GAME_ID = "2025_01_ARI_NO"

NFLVERSE_REPLAY_COLUMNS = (
    "game_id",
    "play_id",
    "order_sequence",
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


@lru_cache(maxsize=1)
def _assert_x11_registered() -> None:
    assert "X-11" in load_experiment_registry(PROJECT_ROOT)


@lru_cache(maxsize=1)
def _cached_registry() -> dict[str, dict[str, Any]]:
    return load_experiment_registry(PROJECT_ROOT)


def _real_game_rows() -> list[dict[str, Any]]:
    if not NFLVERSE_2025.is_file():
        pytest.skip("frozen nflverse 2025 raw object is not present")
    parquet = pytest.importorskip("pyarrow.parquet")
    table = parquet.read_table(
        NFLVERSE_2025,
        columns=list(NFLVERSE_REPLAY_COLUMNS),
        filters=[("game_id", "=", GAME_ID)],
    )
    rows = sorted(
        table.to_pylist(),
        key=lambda row: int(row["order_sequence"]),
    )
    if len(rows) < 2:
        pytest.fail(f"frozen raw object did not contain complete game {GAME_ID}")
    return rows


def _event_for_rows(
    state: nfl.NFLGameState,
    source_row: dict[str, Any],
    successor_rows: tuple[dict[str, Any], ...],
    *,
    sequence: int,
) -> nfl.NFLPlayEvent:
    payload = nfl.nflverse_transition_payload(
        state,
        source_row,
        successor_rows,
        sequence=sequence,
    )
    _assert_x11_registered()
    canonical_refs = {
        "competition_id": "cmp_nfl",
        "game_id": f"game_nflverse_{GAME_ID}",
        "participant_ids": ("participant_ARI", "participant_NO"),
        "venue_event_id": None,
        "market_id": None,
        "outcome_id": None,
        "condition_id": None,
    }
    event_time = {
        "receive_at": "2026-07-23T03:30:50.704643Z",
        "receive_basis": "upstream_exporter",
        "source_at": None,
        "publish_at": None,
        "exchange_at": None,
    }
    raw_parents = tuple(
        EventEnvelopeV0.create(
            envelope_version="v0",
            event_type="raw_observation",
            payload_schema_version="v0",
            source={
                "system": "nflverse",
                "stream": "play_by_play",
                "venue": None,
                "sequence": ordinal,
                "capture_session_id": f"static:{RAW_OBJECT_SHA256}",
                "record_ordinal": ordinal,
            },
            time=event_time,
            canonical_refs=canonical_refs,
            native_refs=(
                {
                    "namespace": "nflverse.play",
                    "native_id": f"{GAME_ID}:{int(row['play_id'])}",
                },
            ),
            lineage={
                "raw_object_hash": RAW_OBJECT_SHA256,
                "raw_record_ordinal": ordinal,
                "parent_event_ids": (),
            },
            experiment_id=None,
            rule_snapshot_ref=None,
            quality_flags=(),
            payload={
                "dataset_id": "DS-NFLVERSE",
                "partition": "season-2025",
                "raw_object_hash": RAW_OBJECT_SHA256,
                "raw_record_ordinal": ordinal,
            },
        )
        for row in (source_row, *successor_rows)
        for ordinal in (int(row["order_sequence"]),)
    )
    normalized = EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="normalized_observation",
        payload_schema_version="v0",
        source={
            "system": "nflverse",
            "stream": "play_by_play.normalized",
            "venue": None,
            "sequence": sequence,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        time=event_time,
        canonical_refs=canonical_refs,
        native_refs=tuple(
            parent.native_refs[0] for parent in raw_parents
        ),
        lineage={
            "raw_object_hash": None,
            "raw_record_ordinal": None,
            "parent_event_ids": tuple(
                parent.event_id for parent in raw_parents
            ),
        },
        experiment_id="X-11",
        rule_snapshot_ref=None,
        quality_flags=(),
        payload=payload,
    )
    with patch(
        "prediction_market.experiments.load_experiment_registry",
        return_value=_cached_registry(),
    ):
        return nfl.event_from_nflverse_envelope(
            normalized,
            program_root=PROJECT_ROOT,
            raw_parents=raw_parents,
        )


def _carries_context(row: dict[str, Any]) -> bool:
    description = row["desc"]
    lifecycle = (
        row["play_type_nfl"] == "COMMENT"
        and isinstance(description, str)
        and (
            description.startswith("The game has been suspended.")
            or description.startswith("The game has resumed.")
        )
    )
    administrative_timeout = (
        row["play_type_nfl"] == "TIMEOUT"
        and row["play_type"] == "no_play"
    )
    return lifecycle or administrative_timeout or row["quarter_end"] == 1


def _successor_window(
    rows: list[dict[str, Any]],
    source_index: int,
) -> tuple[dict[str, Any], ...]:
    if source_index >= len(rows) - 1:
        raise AssertionError("a source row requires an immediate successor")
    if _carries_context(rows[source_index]):
        return (rows[source_index + 1],)

    window: list[dict[str, Any]] = []
    for row in rows[source_index + 1 :]:
        window.append(row)
        if row["posteam"] is not None:
            return tuple(window)
    raise AssertionError("normal source row has no complete causal successor")


def _mapped_scores(
    row: dict[str, Any],
    *,
    after_play: bool,
) -> tuple[int, int]:
    suffix = "_post" if after_play else ""
    posteam_score = row[f"posteam_score{suffix}"]
    defteam_score = row[f"defteam_score{suffix}"]
    if row["posteam"] == row["home_team"]:
        return int(posteam_score), int(defteam_score)
    if row["posteam"] == row["away_team"]:
        return int(defteam_score), int(posteam_score)
    return int(row["total_home_score"]), int(row["total_away_score"])


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _stable_optional_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def test_real_touchdown_and_extra_point_are_scored_on_their_own_rows() -> None:
    rows = _real_game_rows()
    touchdown_index = next(
        index
        for index, row in enumerate(rows[:-2])
        if row["sp"] == 1
        and row["posteam_score_post"] - row["posteam_score"] == 6
        and rows[index + 1]["play_type_nfl"] == "XP_KICK"
    )
    touchdown_row = rows[touchdown_index]
    point_after_row = rows[touchdown_index + 1]
    following_row = rows[touchdown_index + 2]

    state = nfl.state_from_nflverse_row(touchdown_row, sequence=0)
    assert (state.home_score, state.away_score) == _mapped_scores(
        touchdown_row,
        after_play=False,
    )

    touchdown = _event_for_rows(
        state,
        touchdown_row,
        (point_after_row,),
        sequence=1,
    )
    after_touchdown = nfl.reduce(state, touchdown)
    assert touchdown.source_play_id == str(int(touchdown_row["play_id"]))
    assert (after_touchdown.home_score, after_touchdown.away_score) == (
        _mapped_scores(touchdown_row, after_play=True)
    )

    point_after = _event_for_rows(
        after_touchdown,
        point_after_row,
        (following_row,),
        sequence=2,
    )
    after_point = nfl.reduce(after_touchdown, point_after)
    assert point_after.source_play_id == str(int(point_after_row["play_id"]))
    assert (after_point.home_score, after_point.away_score) == _mapped_scores(
        point_after_row,
        after_play=True,
    )


def test_real_timeout_is_not_shifted_to_prior_play_and_carries_context() -> None:
    rows = _real_game_rows()
    timeout_index = next(
        index
        for index, row in enumerate(rows[1:-1], start=1)
        if row["timeout"] == 1 and row["play_type_nfl"] == "TIMEOUT"
    )
    prior_row = rows[timeout_index - 1]
    timeout_row = rows[timeout_index]
    following_row = rows[timeout_index + 1]

    state = nfl.state_from_nflverse_row(prior_row, sequence=0)
    prior_play = _event_for_rows(
        state,
        prior_row,
        (timeout_row, following_row),
        sequence=1,
    )
    assert prior_play.timeout_observed is False
    assert prior_play.carry_forward_context is False
    before_timeout = nfl.reduce(state, prior_play)
    assert (
        before_timeout.possession_team,
        before_timeout.down,
        before_timeout.distance,
        before_timeout.yardline_100,
    ) == (
        following_row["posteam"],
        _optional_int(following_row["down"]),
        _optional_int(following_row["ydstogo"]),
        _optional_int(following_row["yardline_100"]),
    )
    assert before_timeout.context_source_play_id == str(
        int(following_row["play_id"])
    )

    timeout = _event_for_rows(
        before_timeout,
        timeout_row,
        (following_row,),
        sequence=2,
    )
    assert timeout.source_play_id == str(int(timeout_row["play_id"]))
    assert timeout.timeout_observed is True
    assert timeout.timeout_kind == "administrative"
    assert timeout.timeout_observed_team == timeout_row["timeout_team"]
    assert timeout.carry_forward_context is True

    after_timeout = nfl.reduce(before_timeout, timeout)
    assert after_timeout.home_timeouts_remaining == int(
        timeout_row["home_timeouts_remaining"]
    )
    assert after_timeout.away_timeouts_remaining == int(
        timeout_row["away_timeouts_remaining"]
    )
    assert (
        after_timeout.possession_team,
        after_timeout.down,
        after_timeout.distance,
        after_timeout.yardline_100,
    ) == (
        before_timeout.possession_team,
        before_timeout.down,
        before_timeout.distance,
        before_timeout.yardline_100,
    )


def _replay_once() -> tuple[str, int]:
    rows = _real_game_rows()
    state = nfl.state_from_nflverse_row(rows[0], sequence=0)

    for source_index, pre_row in enumerate(rows[:-1]):
        sequence = source_index + 1
        successor_rows = _successor_window(rows, source_index)
        post_row = successor_rows[0]
        prior_state = state
        event = _event_for_rows(
            state,
            pre_row,
            successor_rows,
            sequence=sequence,
        )
        state = nfl.reduce(state, event)
        assert event.source_play_id == str(int(pre_row["play_id"]))
        assert state.source_play_id == str(int(post_row["play_id"]))
        assert event.source_order_sequence == int(pre_row["order_sequence"])
        assert state.source_order_sequence == int(post_row["order_sequence"])
        assert state.sequence == sequence
        assert state.last_event_id == event.event_id
        assert (state.home_score, state.away_score) == _mapped_scores(
            pre_row,
            after_play=True,
        )

        pre_timeout = pre_row["timeout"] == 1
        post_timeout = post_row["timeout"] == 1
        pre_admin_timeout = (
            pre_timeout and pre_row["play_type_nfl"] == "TIMEOUT"
        )
        post_admin_timeout = (
            post_timeout and post_row["play_type_nfl"] == "TIMEOUT"
        )
        assert event.timeout_observed is pre_timeout
        if pre_timeout:
            assert event.timeout_observed_team == pre_row["timeout_team"]
            assert event.timeout_kind == (
                "administrative" if pre_admin_timeout else "play_attached"
            )
        expected_timeouts = (
            (
                int(pre_row["home_timeouts_remaining"]),
                int(pre_row["away_timeouts_remaining"]),
            )
            if pre_timeout
            else (
                state.home_timeouts_remaining,
                state.away_timeouts_remaining,
            )
        )
        assert (
            state.home_timeouts_remaining,
            state.away_timeouts_remaining,
        ) == expected_timeouts

        expected_carry = _carries_context(pre_row)
        assert event.carry_forward_context is expected_carry
        if expected_carry:
            assert (
                state.drive_id,
                state.play_clock_seconds,
                state.possession_team,
                state.down,
                state.distance,
                state.yardline_100,
                state.goal_to_go,
            ) == (
                prior_state.drive_id,
                prior_state.play_clock_seconds,
                prior_state.possession_team,
                prior_state.down,
                prior_state.distance,
                prior_state.yardline_100,
                prior_state.goal_to_go,
            )
        else:
            context_row = successor_rows[-1]
            assert (
                state.drive_id,
                state.play_clock_seconds,
                state.possession_team,
                state.down,
                state.distance,
                state.yardline_100,
                state.goal_to_go,
            ) == (
                _stable_optional_scalar(context_row["fixed_drive"]),
                _optional_int(context_row["play_clock"]),
                context_row["posteam"],
                _optional_int(context_row["down"]),
                (
                    _optional_int(context_row["ydstogo"])
                    if context_row["down"] is not None
                    else None
                ),
                _optional_int(context_row["yardline_100"]),
                bool(context_row["goal_to_go"]),
            )

        clock_row = (
            pre_row
            if (
                pre_admin_timeout
                or (
                    post_admin_timeout
                    and post_row["qtr"] == pre_row["qtr"]
                )
            )
            else post_row
        )
        assert (
            state.period,
            state.period_seconds_remaining,
            state.game_seconds_remaining,
        ) == (
            int(clock_row["qtr"]),
            int(clock_row["quarter_seconds_remaining"]),
            int(clock_row["game_seconds_remaining"]),
        )

    return canonical_state_sha256(state), len(rows) - 1


def test_real_snapshot_flags_are_committed_only_when_transition_verifies_them() -> None:
    rows = _real_game_rows()
    state = nfl.state_from_nflverse_row(rows[38], sequence=0)

    first_down_before_quarter_end = _event_for_rows(
        state,
        rows[38],
        _successor_window(rows, 38),
        sequence=39,
    )
    assert rows[38]["first_down"] == 1
    assert rows[39]["desc"] == "END QUARTER 1"
    assert first_down_before_quarter_end.first_down is False


def test_future_drive_result_cannot_change_a_normalized_transition() -> None:
    rows = _real_game_rows()
    pre_row = rows[20]
    successors = _successor_window(rows, 20)
    state = nfl.state_from_nflverse_row(pre_row, sequence=0)
    baseline = _event_for_rows(
        state,
        pre_row,
        successors,
        sequence=21,
    )
    mutated_pre = {**pre_row, "series_result": "Future impossible outcome"}
    mutated_successors = tuple(
        {
            **row,
            "series_result": "Another future outcome",
            "play_type": "future play must not leak",
        }
        for row in successors
    )

    assert "series_result" in nfl.NFLVERSE_LEAKAGE_FIELDS
    assert _event_for_rows(
        state,
        mutated_pre,
        mutated_successors,
        sequence=21,
    ) == baseline


def test_real_complete_game_replays_twice_to_identical_hash() -> None:
    first_hash, first_steps = _replay_once()
    second_hash, second_steps = _replay_once()

    assert first_steps == second_steps == 181
    assert first_hash == second_hash


def test_adapter_requires_a_fully_bound_event_envelope() -> None:
    rows = _real_game_rows()
    state = nfl.state_from_nflverse_row(rows[0], sequence=0)
    event = _event_for_rows(
        state,
        rows[0],
        _successor_window(rows, 0),
        sequence=1,
    )
    assert event.source_play_id == str(int(rows[0]["play_id"]))

    with pytest.raises(TypeError, match="EventEnvelopeV0"):
        nfl.event_from_nflverse_envelope(  # type: ignore[arg-type]
            rows[0],
            program_root=PROJECT_ROOT,
            raw_parents=(),
        )
