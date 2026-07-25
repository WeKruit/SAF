"""Completion-oriented audit package for the first DAL–DET reconstruction."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prediction_market.contracts import canonical_json_bytes, canonical_sha256
from prediction_market.sports.nfl_dal_det_event_market import (
    build_dal_det_event_market_source_time_timeline,
)
from prediction_market.sports.nfl_dal_det_gamebook_crosscheck import (
    build_dal_det_gamebook_crosscheck,
)
from prediction_market.sports.nfl_dal_det_personnel import (
    build_dal_det_event_personnel_timeline,
)
from prediction_market.sports.nfl_game_replay import replay_frozen_nfl_game


class DALDETReviewError(ValueError):
    """The single-game review cannot prove its declared first-layer gates."""


def build_dal_det_first_layer_review(
    *, program_root: str | Path, market_mapping: Mapping[str, object]
) -> dict[str, Any]:
    """Produce the compact evidence handoff for reconstruction before causality."""

    root = Path(program_root).resolve()
    replay = replay_frozen_nfl_game(program_root=root, native_game_id="2025_14_DAL_DET")
    personnel = build_dal_det_event_personnel_timeline(program_root=root)
    event_market = build_dal_det_event_market_source_time_timeline(
        program_root=root, mapping=market_mapping
    )
    gamebook = build_dal_det_gamebook_crosscheck(program_root=root)
    final_state = replay.steps[-1]["next_state"]
    if (final_state["away_score"], final_state["home_score"]) != (30, 44):
        raise DALDETReviewError("frozen replay final score is not DAL 30 / DET 44")
    if personnel["join"]["full_11v11_rows"] != personnel["join"]["joined_rows"]:
        raise DALDETReviewError("not all joined rows have 11v11 coverage")
    if event_market["merge_validation"]["status"] != "PASS_SOURCE_TIME_CHRONOLOGY_ONLY":
        raise DALDETReviewError("source-time chronology gate did not pass")
    if not gamebook["comparison"]["all_official_score_milestones_match"]:
        raise DALDETReviewError("official scorecard cross-check did not pass")
    report: dict[str, Any] = {
        "artifact_type": "nfl_dal_det_first_layer_review",
        "artifact_version": "v0",
        "canonical_game_id": replay.canonical_game_id,
        "scope": "game replay + full observed player participation + market source-time chronology",
        "inputs": {
            "replay_trace_sha256": replay.trace_sha256,
            "event_personnel_artifact_sha256": personnel["artifact_sha256"],
            "event_market_rows_sha256": event_market["rows_sha256"],
            "official_scorecard_crosscheck_sha256": gamebook["artifact_sha256"],
        },
        "verified": {
            "final_score": {"DAL": 30, "DET": 44},
            "replay": {
                "source_rows": replay.source_row_count,
                "transitions": replay.transition_count,
                "deterministic_two_run_hash": replay.trace_sha256,
                "timestamped_source_rows": replay.source_time_present_count,
            },
            "play_personnel": {
                "join_key": personnel["join"]["join_key"],
                "full_11v11_play_rows": personnel["join"]["full_11v11_rows"],
                "observed_pbp_rows_without_participation": personnel["join"]["administrative_or_nonparticipation_pbp_rows"],
            },
            "market_source_time": event_market["counts"],
            "official_scorecard": {
                "final_score_match": gamebook["comparison"]["final_score_match"],
                "all_completed_score_milestones_match": gamebook["comparison"]["all_official_score_milestones_match"],
            },
        },
        "gates": {
            "G_game_state_replay": "PASS",
            "P_play_personnel_join": "PASS",
            "M_market_history_replay": "PASS_SOURCE_OBSERVATIONS_ONLY",
            "A_source_time_chronological_merge": "PASS",
            "O_official_scorecard_crosscheck": "PASS_NARROW",
            "R_event_to_market_reaction": "BLOCKED",
        },
        "blocked_for_next_layer": [
            "No game-feed local_received_time or measured provider-to-local latency distribution in 2025 historical PBP.",
            "No measured cross-source clock-error bound between game event source time and venue timestamps.",
            "Polymarket evidence for this game is trades only, not historical bid/ask or L2.",
            "Kalshi BBO evidence is one-minute interval-end candles, not event-time L1/L2.",
            "Official Gamebook is licence-blocked from program ingestion; only a narrow external scorecard cross-check is recorded.",
        ],
        "no_go_preserved": {
            "reaction_or_alpha_claim": False,
            "executable_bid_ask_claim": False,
            "queue_fill_claim": False,
            "real_money_action": False,
        },
    }
    report["artifact_sha256"] = canonical_sha256({key: value for key, value in report.items() if key != "artifact_sha256"})
    return report


def write_dal_det_first_layer_review(report: Mapping[str, object], *, output_path: str | Path) -> None:
    path = Path(output_path)
    if path.suffix != ".json":
        raise DALDETReviewError("review output must be JSON")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical_json_bytes(dict(report)) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = ["DALDETReviewError", "build_dal_det_first_layer_review", "write_dal_det_first_layer_review"]
