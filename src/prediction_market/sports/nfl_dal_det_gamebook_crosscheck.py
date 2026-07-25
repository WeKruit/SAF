"""Independent official-scorecard cross-check for the DAL–DET replay.

The NFL Gamebook is deliberately not ingested as program raw: its displayed
copyright notice blocks this research pipeline pending written permission and
the current direct PDF endpoint is not retrievable.  This module therefore
records a narrow external-reference check only: the official post-score path
must be an ordered subsequence of the frozen PBP score path.  It is not a
license workaround and does not make the gamebook an experiment input.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prediction_market.contracts import canonical_json_bytes, canonical_sha256
from prediction_market.sports.nfl_dal_det_personnel import (
    build_dal_det_event_personnel_timeline,
)


_GAMEBOOK_URL = "https://static.www.nfl.com/image/upload/v1764910030/gamecenter/f877de0a-311e-11f0-b670-ae1250fadad1.pdf"
# Official Gamebook scoring plays, represented as (Dallas, Detroit) after the
# completed scoring unit.  The PBP trace also has intermediate pre-PAT states;
# those are intentionally not required here.
_OFFICIAL_POST_SCORE_PATH = (
    (0, 3), (3, 3), (3, 10), (6, 10), (6, 17), (9, 17), (9, 20),
    (9, 27), (16, 27), (19, 27), (19, 30), (27, 30), (27, 37),
    (30, 37), (30, 44),
)


class DALDETGamebookCrosscheckError(ValueError):
    """The frozen PBP score path contradicts the independent scorecard."""


def _score_path(rows: Sequence[Mapping[str, object]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    previous: tuple[int, int] | None = None
    for row in rows:
        state = row.get("state")
        if not isinstance(state, Mapping):
            raise DALDETGamebookCrosscheckError("event/personnel row has no state")
        score = (int(state["total_away_score"]), int(state["total_home_score"]))
        if score != previous:
            path.append(score)
            previous = score
    return path


def _subsequence_positions(path: Sequence[tuple[int, int]], expected: Sequence[tuple[int, int]]) -> list[int]:
    positions: list[int] = []
    cursor = 0
    for score in expected:
        while cursor < len(path) and path[cursor] != score:
            cursor += 1
        if cursor == len(path):
            raise DALDETGamebookCrosscheckError(f"official score milestone {score} is missing from frozen PBP")
        positions.append(cursor)
        cursor += 1
    return positions


def build_dal_det_gamebook_crosscheck(*, program_root: str | Path) -> dict[str, Any]:
    """Validate official final score and scoring progression without ingestion."""

    timeline = build_dal_det_event_personnel_timeline(program_root=program_root)
    score_path = _score_path(timeline["rows"])
    positions = _subsequence_positions(score_path, _OFFICIAL_POST_SCORE_PATH)
    report: dict[str, Any] = {
        "artifact_type": "nfl_dal_det_official_scorecard_crosscheck",
        "artifact_version": "v0",
        "canonical_game_id": timeline["canonical_game_id"],
        "inputs": {"event_personnel_artifact_sha256": timeline["artifact_sha256"]},
        "official_reference": {
            "source_url": _GAMEBOOK_URL,
            "source_kind": "NFL official Gamebook PDF",
            "license_review_id": "O-010",
            "license_status": "blocked",
            "raw_ingested": False,
            "direct_download_status_as_of_2026_07_24": "HTTP_200_NOT_INGESTED",
            "reason_not_ingested": "Copyright notice limits the material to media coverage and the program has no written permission.",
        },
        "comparison": {
            "method": "official completed scoring-unit path must be an ordered subsequence of frozen PBP score transitions",
            "official_post_score_path": [list(score) for score in _OFFICIAL_POST_SCORE_PATH],
            "frozen_pbp_score_path": [list(score) for score in score_path],
            "official_positions_in_frozen_path": positions,
            "final_score_match": score_path[-1] == _OFFICIAL_POST_SCORE_PATH[-1],
            "all_official_score_milestones_match": len(positions) == len(_OFFICIAL_POST_SCORE_PATH),
            "event_by_event_gamebook_match": False,
        },
        "evidence_boundary": {
            "independent_scorecard_crosscheck": True,
            "official_document_is_not_a_program_raw_input": True,
            "full_official_event_by_event_parse": False,
            "market_time_alignment": False,
        },
    }
    report["artifact_sha256"] = canonical_sha256({key: value for key, value in report.items() if key != "artifact_sha256"})
    return report


def write_dal_det_gamebook_crosscheck(report: Mapping[str, object], *, output_path: str | Path) -> None:
    path = Path(output_path)
    if path.suffix != ".json":
        raise DALDETGamebookCrosscheckError("cross-check output must be JSON")
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


__all__ = [
    "DALDETGamebookCrosscheckError",
    "build_dal_det_gamebook_crosscheck",
    "write_dal_det_gamebook_crosscheck",
]
