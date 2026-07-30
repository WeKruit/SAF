"""Regulation-only, source-linked NFL finalized information episodes.

This module consumes only the verified X-13 Exact-153 Facts V4 publication.
It deliberately refuses to infer episode linkage from descriptions, adjacency,
or market data.  Rows are joined only by explicit ``score_sequence_id`` or
``adjudication_sequence_id`` values already present in the frozen fact layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import resource
import tempfile
import time
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


EXPERIMENT_ID: Final[str] = "X-15"
SCHEMA_VERSION: Final[str] = "FinalizedEpisodeV2"
INFORMATION_ANCHOR_SCHEMA: Final[str] = "ProvisionalInformationAnchorV2"
CONSTITUENT_SCHEMA: Final[str] = "FinalizedEpisodeConstituentV2"
RECONCILIATION_SCHEMA: Final[str] = "FinalizedEpisodeReconciliationV2"
MANIFEST_SCHEMA: Final[str] = "nfl_x15_finalized_episode_batch_v2"
PUBLICATION_ID: Final[str] = "finalized-episodes-regulation-v2"
BUILDER_VERSION: Final[str] = "nfl_x15_finalized_episodes_v2"
CLAIM_BOUNDARY: Final[str] = (
    "REGULATION_SPORTS_INFORMATION_EPISODES_ONLY; no market data, holdout "
    "reaction, latency, causality, execution, or alpha claim"
)
SOURCE_SCOPE: Final[str] = "VERIFIED_EXACT_153_FACTS_V4_ONLY"

DEFAULT_FACTS_BATCH: Final[Path] = Path(
    "artifacts/market-observation/nfl/x13/exact-153-facts-v4/"
    "batches/manifests/sha256/5d/"
    "5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757"
    ".batch-index.json"
)
DEFAULT_FACTS_BATCH_FILE_SHA256: Final[str] = (
    "sha256:5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757"
)
DEFAULT_OUTPUT_ROOT: Final[Path] = Path(
    "artifacts/market-observation/nfl/x15"
)

_FACTS_BATCH_SCHEMA: Final[str] = "nfl_x13_exact153_fact_batch_index_v4"
_FACTS_GAME_SCHEMA: Final[str] = (
    "nfl_x13_exact153_single_game_fact_manifest_v4"
)
_FACTS_PUBLICATION_ID: Final[str] = "exact-153-facts-v4"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GAME_END_NO_TRY_TAGS: Final[frozenset[str]] = frozenset(
    {"GAME_END_NO_TRY_CONFIRMED", "NO_TRY_REQUIRED_GAME_END"}
)
_NULLIFIED_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {"DELETED", "NULLIFIED_NO_PLAY"}
)
_ADJUDICATION_TAGS: Final[frozenset[str]] = frozenset(
    {
        "PENALTY",
        "PENALTY_ACCEPTED",
        "PENALTY_ACCEPTED_NO_PLAY",
        "PENALTY_DECLINED",
        "PENALTY_OFFSETTING",
        "REVIEW",
        "REVERSAL",
    }
)
_TURNOVER_TAGS: Final[frozenset[str]] = frozenset(
    {
        "TURNOVER",
        "INTERCEPTION",
        "LOST_FUMBLE",
        "TURNOVER_ON_DOWNS",
        "MUFFED_PUNT",
        "MUFFED_KICKOFF",
    }
)
_TOUCHDOWN_TAGS: Final[frozenset[str]] = frozenset(
    {
        "TOUCHDOWN",
        "PASS_TOUCHDOWN",
        "RUSH_TOUCHDOWN",
        "RETURN_TOUCHDOWN",
    }
)

_REQUIRED_EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "event_id",
    "raw_play_id",
    "play_id",
    "atomic_information_episode_id",
    "score_sequence_id",
    "adjudication_sequence_id",
    "information_status",
    "stage_b_information_event_eligible",
    "final_sports_outcome_eligible",
    "source_interval_start",
    "source_interval_end",
    "order_sequence",
    "quarter",
    "home_team",
    "away_team",
    "actor_team",
    "beneficiary_team",
    "pre_home_score",
    "pre_away_score",
    "post_home_score",
    "post_away_score",
    "primary_action",
    "outcome_tags",
    "yards_gained",
    "return_yards",
    "review_result",
    "row_disposition",
    "known_at",
    "source_hashes",
    "pbp_source_sha256",
    "participation_source_sha256",
)

_EPISODE_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "claim_boundary",
    "game_id",
    "episode_id",
    "episode_status",
    "episode_type",
    "constituent_count",
    "score_sequence_id",
    "adjudication_sequence_id",
    "quarter",
    "first_seen_interval_start",
    "first_seen_interval_end",
    "final_known_interval_start",
    "final_known_interval_end",
    "finalized_anchor",
    "finalized_anchor_semantics",
    "finalization_gap_seconds",
    "stage_a_eligible",
    "residual_diagnostic_eligible",
    "exclusion_reason",
    "home_team",
    "away_team",
    "final_beneficiary_team",
    "final_pre_home_score",
    "final_pre_away_score",
    "final_post_home_score",
    "final_post_away_score",
    "final_score_points",
    "final_turnover",
    "final_outcome_tags",
    "source_hashes",
)

_INFORMATION_ANCHOR_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "claim_boundary",
    "game_id",
    "information_anchor_id",
    "episode_id",
    "event_id",
    "raw_play_id",
    "order_sequence",
    "constituent_role",
    "score_sequence_id",
    "adjudication_sequence_id",
    "source_interval_start",
    "source_interval_end",
    "information_anchor",
    "information_anchor_semantics",
    "provisional_known_at",
    "provisional_primary_event_id",
    "provisional_primary_action",
    "provisional_outcome_tags",
    "provisional_actor_team",
    "provisional_beneficiary_team",
    "provisional_score_points_observed",
    "provisional_turnover_observed",
    "provisional_yards_gained",
    "provisional_return_yards",
    "provisional_feature_support_status",
    "provisional_feature_support_reason",
    "provisional_snapshot_sha256",
    "canonical_stage_b_information_event_eligible",
    "primary_selection_eligible",
    "censor_boundary_eligible",
    "source_hashes",
    "pbp_source_sha256",
)

_CONSTITUENT_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "claim_boundary",
    "game_id",
    "event_id",
    "raw_play_id",
    "order_sequence",
    "episode_id",
    "mapping_status",
    "constituent_role",
    "exclusion_reason",
    "score_sequence_id",
    "adjudication_sequence_id",
    "source_interval_start",
    "source_interval_end",
    "final_outcome_tags",
    "final_turnover",
    "final_score_points",
    "pbp_source_sha256",
)

_RECONCILIATION_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "claim_boundary",
    "game_id",
    "raw_row_count",
    "mapped_row_count",
    "excluded_row_count",
    "episode_count",
    "information_anchor_count",
    "stage_a_eligible_episode_count",
    "residual_diagnostic_eligible_episode_count",
    "primary_selection_eligible_anchor_count",
    "censor_boundary_eligible_anchor_count",
    "unresolved_adjudication_episode_count",
    "data_gap_episode_count",
    "nullified_episode_count",
    "ot_excluded_row_count",
    "administrative_excluded_row_count",
    "raw_rows_silently_dropped",
    "duplicate_event_mapping_count",
    "publication_gate",
)


class FinalizedEpisodeError(RuntimeError):
    """A source, episode, reconciliation, or publication invariant failed."""


@dataclass(frozen=True, slots=True)
class FinalizedEpisodeTables:
    episodes: pd.DataFrame
    information_anchors: pd.DataFrame
    constituents: pd.DataFrame
    reconciliation: pd.DataFrame


@dataclass(frozen=True, slots=True)
class FinalizedEpisodePublication:
    manifest_path: Path
    manifest_sha256: str
    bundle_sha256: str
    game_count: int
    raw_row_count: int
    information_anchor_count: int
    episode_count: int
    stage_a_eligible_episode_count: int
    residual_diagnostic_eligible_episode_count: int
    primary_selection_eligible_anchor_count: int
    censor_boundary_eligible_anchor_count: int
    unresolved_adjudication_episode_count: int
    data_gap_episode_count: int
    nullified_episode_count: int
    ot_excluded_row_count: int
    raw_rows_silently_dropped: int
    runtime_seconds: float
    peak_rss_mib: float


def _present(value: object) -> bool:
    if value is None or value is pd.NA:
        return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return True
    return not bool(result) if isinstance(result, (bool, np.bool_)) else True


def _text(value: object) -> str | None:
    if not _present(value):
        return None
    result = str(value).strip()
    return result or None


def _integer(value: object) -> int | None:
    if not _present(value) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _number(value: object) -> float | None:
    if not _present(value) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if not _present(value):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _tags(value: object) -> tuple[str, ...]:
    if not _present(value):
        return ()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise FinalizedEpisodeError("outcome_tags is not valid JSON") from exc
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        parsed = value
    else:
        raise FinalizedEpisodeError("outcome_tags must be a JSON array")
    if not isinstance(parsed, Sequence) or isinstance(parsed, str):
        raise FinalizedEpisodeError("outcome_tags must be a JSON array")
    output: list[str] = []
    for item in parsed:
        text = _text(item)
        if text is None:
            raise FinalizedEpisodeError("outcome_tags contains an empty value")
        output.append(text.upper())
    return tuple(dict.fromkeys(output))


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    raise FinalizedEpisodeError(
        f"value is not canonical JSON: {type(value).__name__}"
    )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_sha(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _canonical_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(key): _canonical_value(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise FinalizedEpisodeError(f"{field} must be a SHA-256 digest")
    return value


def _safe_path(root: Path, relative: str | Path, *, field: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise FinalizedEpisodeError(f"{field} escapes its declared root")
    return candidate


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FinalizedEpisodeError(
                f"content-addressed object collision: {path}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FinalizedEpisodeError(
                    f"content-addressed publish race: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _is_adjudication_row(row: Mapping[str, object], tags: set[str]) -> bool:
    action = (_text(row.get("primary_action")) or "").upper()
    return (
        action == "PENALTY"
        or _text(row.get("review_result")) is not None
        or bool(tags & _ADJUDICATION_TAGS)
        or (_text(row.get("information_status")) or "FINAL").upper()
        not in {"FINAL"}
    )


def _is_row_nullified(row: Mapping[str, object]) -> bool:
    action = (_text(row.get("primary_action")) or "").upper()
    if action == "TIMEOUT":
        return False
    disposition = (_text(row.get("row_disposition")) or "").upper()
    information_status = (
        _text(row.get("information_status")) or "FINAL"
    ).upper()
    return (
        disposition in _NULLIFIED_DISPOSITIONS
        or information_status in {"NULLIFIED", "DELETED"}
    )


def _is_score_sequence_member(
    row: Mapping[str, object], tags: set[str]
) -> bool:
    if _text(row.get("score_sequence_id")) is None:
        return False
    action = (_text(row.get("primary_action")) or "").upper()
    return action != "KICKOFF" or bool(tags & _TOUCHDOWN_TAGS)


def _anchor_gap_seconds(
    provisional_anchor: str | None, finalized_anchor: str | None
) -> float | None:
    if provisional_anchor is None or finalized_anchor is None:
        return None
    try:
        provisional = pd.Timestamp(provisional_anchor)
        finalized = pd.Timestamp(finalized_anchor)
    except (TypeError, ValueError) as exc:
        raise FinalizedEpisodeError("episode anchor is not a timestamp") from exc
    gap = float((finalized - provisional).total_seconds())
    if not math.isfinite(gap):
        raise FinalizedEpisodeError("episode finalization gap is not finite")
    return gap


def _observed_score_points(row: Mapping[str, object]) -> int | None:
    pre_home = _integer(row.get("pre_home_score"))
    pre_away = _integer(row.get("pre_away_score"))
    post_home = _integer(row.get("post_home_score"))
    post_away = _integer(row.get("post_away_score"))
    if None in {pre_home, pre_away, post_home, post_away}:
        return None
    assert pre_home is not None
    assert pre_away is not None
    assert post_home is not None
    assert post_away is not None
    return max(0, post_home - pre_home) + max(0, post_away - pre_away)


def _provisional_feature_support(
    row: Mapping[str, object],
    tags: set[str],
    *,
    provisional_anchor: str | None,
) -> tuple[str, str | None]:
    review_or_reversal = (
        _text(row.get("review_result")) is not None
        or bool(
            tags
            & {
                "REVIEW",
                "REVIEWED",
                "REVERSAL",
                "REVERSED",
                "REVIEW_REVERSED",
            }
        )
    )
    if review_or_reversal:
        return (
            "PRIMARY_FEATURE_SUPPORT_UNPROVEN",
            "REVIEW_REVISION_CHRONOLOGY_UNAVAILABLE",
        )
    known_at = _text(row.get("known_at"))
    if known_at is None:
        return (
            "PRIMARY_FEATURE_SUPPORT_UNPROVEN",
            "PROVISIONAL_KNOWN_AT_MISSING",
        )
    if provisional_anchor is None:
        return (
            "PRIMARY_FEATURE_SUPPORT_UNPROVEN",
            "PROVISIONAL_FIRST_SEEN_INTERVAL_MISSING",
        )
    try:
        known_timestamp = pd.Timestamp(known_at)
        anchor_timestamp = pd.Timestamp(provisional_anchor)
    except (TypeError, ValueError):
        return (
            "PRIMARY_FEATURE_SUPPORT_UNPROVEN",
            "PROVISIONAL_TIME_PARSE_FAILED",
        )
    if known_timestamp > anchor_timestamp:
        return (
            "PRIMARY_FEATURE_SUPPORT_UNPROVEN",
            "PROVISIONAL_KNOWN_AT_AFTER_FIRST_SEEN_ANCHOR",
        )
    return "SUPPORTED", None


def _peak_rss_mib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024


def _source_hashes(rows: Sequence[Mapping[str, object]]) -> str:
    hashes: set[str] = set()
    for row in rows:
        for field in ("pbp_source_sha256", "participation_source_sha256"):
            value = _text(row.get(field))
            if value is not None:
                hashes.add(_require_sha(value, field=field))
        raw = row.get("source_hashes")
        if _present(raw):
            try:
                parsed = json.loads(str(raw)) if isinstance(raw, str) else raw
            except json.JSONDecodeError as exc:
                raise FinalizedEpisodeError(
                    "source_hashes is not valid JSON"
                ) from exc
            if not isinstance(parsed, Sequence) or isinstance(parsed, str):
                raise FinalizedEpisodeError("source_hashes must be an array")
            for value in parsed:
                hashes.add(_require_sha(value, field="source_hashes[]"))
    return json.dumps(sorted(hashes), separators=(",", ":"))


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _constituent_role(row: Mapping[str, object], tags: set[str]) -> str:
    action = (_text(row.get("primary_action")) or "").upper()
    if tags & _TOUCHDOWN_TAGS:
        return "TOUCHDOWN"
    if action == "TRY":
        return "TRY"
    if _is_adjudication_row(row, tags):
        return "ADJUDICATION"
    return "PRIMARY"


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _build_game(
    frame: pd.DataFrame,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    frame = frame.sort_values(
        ["order_sequence", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    game_ids = frame["game_id"].dropna().astype(str).unique().tolist()
    if len(game_ids) != 1:
        raise FinalizedEpisodeError("a game partition must contain one game_id")
    game_id = game_ids[0]
    if frame["event_id"].isna().any() or frame["event_id"].astype(str).duplicated().any():
        raise FinalizedEpisodeError(f"duplicate or missing event_id: {game_id}")
    if frame["order_sequence"].map(_integer).isna().any():
        raise FinalizedEpisodeError(f"invalid order_sequence: {game_id}")

    rows = frame.to_dict("records")
    constituents: list[dict[str, object]] = []
    information_anchors: list[dict[str, object]] = []
    regulation_indices: list[int] = []
    ot_excluded = 0
    admin_excluded = 0
    for index, row in enumerate(rows):
        quarter = _integer(row.get("quarter"))
        if quarter is not None and quarter >= 5:
            ot_excluded += 1
            constituents.append(
                {
                    "schema_version": CONSTITUENT_SCHEMA,
                    "claim_boundary": CLAIM_BOUNDARY,
                    "game_id": game_id,
                    "event_id": str(row["event_id"]),
                    "raw_play_id": _text(row.get("raw_play_id")),
                    "order_sequence": _integer(row.get("order_sequence")),
                    "episode_id": None,
                    "mapping_status": "EXCLUDED",
                    "constituent_role": "EXCLUDED",
                    "exclusion_reason": "OT_QUARTER_GE_5",
                    "score_sequence_id": _text(row.get("score_sequence_id")),
                    "adjudication_sequence_id": _text(
                        row.get("adjudication_sequence_id")
                    ),
                    "source_interval_start": _text(
                        row.get("source_interval_start")
                    ),
                    "source_interval_end": _text(
                        row.get("source_interval_end")
                    ),
                    "final_outcome_tags": "[]",
                    "final_turnover": False,
                    "final_score_points": 0,
                    "pbp_source_sha256": _require_sha(
                        row.get("pbp_source_sha256"),
                        field="pbp_source_sha256",
                    ),
                }
            )
            continue
        action = (_text(row.get("primary_action")) or "").upper()
        tags = set(_tags(row.get("outcome_tags")))
        administrative = (
            action == "ADMIN"
            and not _is_adjudication_row(row, tags)
            and not _is_row_nullified(row)
        )
        if administrative:
            admin_excluded += 1
            constituents.append(
                {
                    "schema_version": CONSTITUENT_SCHEMA,
                    "claim_boundary": CLAIM_BOUNDARY,
                    "game_id": game_id,
                    "event_id": str(row["event_id"]),
                    "raw_play_id": _text(row.get("raw_play_id")),
                    "order_sequence": _integer(row.get("order_sequence")),
                    "episode_id": None,
                    "mapping_status": "EXCLUDED",
                    "constituent_role": "EXCLUDED",
                    "exclusion_reason": "ADMINISTRATIVE_NON_PLAY",
                    "score_sequence_id": _text(row.get("score_sequence_id")),
                    "adjudication_sequence_id": _text(
                        row.get("adjudication_sequence_id")
                    ),
                    "source_interval_start": _text(
                        row.get("source_interval_start")
                    ),
                    "source_interval_end": _text(
                        row.get("source_interval_end")
                    ),
                    "final_outcome_tags": "[]",
                    "final_turnover": False,
                    "final_score_points": 0,
                    "pbp_source_sha256": _require_sha(
                        row.get("pbp_source_sha256"),
                        field="pbp_source_sha256",
                    ),
                }
            )
            continue
        regulation_indices.append(index)

    local = {global_index: local_index for local_index, global_index in enumerate(regulation_indices)}
    sets = _DisjointSet(len(regulation_indices))
    score_groups: dict[str, list[int]] = {}
    adjudication_groups: dict[str, list[int]] = {}
    for global_index in regulation_indices:
        row = rows[global_index]
        tags = set(_tags(row.get("outcome_tags")))
        score_id = _text(row.get("score_sequence_id"))
        if score_id is not None and _is_score_sequence_member(row, tags):
            score_groups.setdefault(score_id, []).append(global_index)
        adjudication_id = _text(row.get("adjudication_sequence_id"))
        if adjudication_id is not None:
            adjudication_groups.setdefault(adjudication_id, []).append(global_index)
    for group in (*score_groups.values(), *adjudication_groups.values()):
        for left, right in zip(group, group[1:]):
            sets.union(local[left], local[right])
    components: dict[int, list[int]] = {}
    for global_index in regulation_indices:
        components.setdefault(sets.find(local[global_index]), []).append(global_index)

    episodes: list[dict[str, object]] = []
    for component in sorted(
        components.values(),
        key=lambda members: (
            _integer(rows[members[0]].get("order_sequence")) or -1,
            str(rows[members[0]]["event_id"]),
        ),
    ):
        component.sort(
            key=lambda index: (
                _integer(rows[index].get("order_sequence")) or -1,
                str(rows[index]["event_id"]),
            )
        )
        members = [rows[index] for index in component]
        member_tags = [set(_tags(row.get("outcome_tags"))) for row in members]
        score_ids = {
            value
            for member, tags in zip(members, member_tags, strict=True)
            if (
                (value := _text(member.get("score_sequence_id"))) is not None
                and _is_score_sequence_member(member, tags)
            )
        }
        adjudication_ids = {
            value
            for value in (
                _text(row.get("adjudication_sequence_id")) for row in members
            )
            if value is not None
        }
        has_touchdown = any(tags & _TOUCHDOWN_TAGS for tags in member_tags)
        has_try = any(
            (_text(row.get("primary_action")) or "").upper() == "TRY"
            and not _is_row_nullified(row)
            for row in members
        )
        explicit_no_try_evidence = any(
            tags & _GAME_END_NO_TRY_TAGS for tags in member_tags
        )
        adjudication_related = any(
            _is_adjudication_row(row, tags)
            for row, tags in zip(members, member_tags, strict=True)
        )
        stable_adjudication_link = (
            bool(score_ids) and len(members) > 1
        ) or (
            bool(adjudication_ids) and len(members) > 1
        )
        has_reversal = any(
            (_text(row.get("review_result")) or "").casefold() == "reversed"
            or (
                (_text(row.get("information_status")) or "").upper()
                == "REVERSED"
            )
            for row in members
        )
        semantic_pairs = [
            (row, tags)
            for row, tags in zip(members, member_tags, strict=True)
            if not _is_row_nullified(row)
        ]
        all_nullified = not semantic_pairs

        status = "FINAL"
        exclusion_reason: str | None = None
        if has_reversal and not stable_adjudication_link:
            status = "DATA_GAP"
            exclusion_reason = "UNLINKED_REVERSAL"
        elif all_nullified:
            status = "NULLIFIED"
            exclusion_reason = "EXPLICIT_NULLIFIED_OR_NO_PLAY"
        elif adjudication_related and not stable_adjudication_link:
            status = "UNRESOLVED_ADJUDICATION_CHAIN"
            exclusion_reason = "UNRESOLVED_ADJUDICATION_CHAIN"
        elif len(score_ids) > 1 or len(adjudication_ids) > 1:
            status = "DATA_GAP"
            exclusion_reason = "CONFLICTING_STRUCTURED_LINKAGE"
        elif score_ids and not has_touchdown:
            status = "DATA_GAP"
            exclusion_reason = "SCORE_SEQUENCE_WITHOUT_TOUCHDOWN"
        elif has_touchdown and not has_try and not explicit_no_try_evidence:
            status = "DATA_GAP"
            exclusion_reason = "MISSING_TRY_OR_GAME_END_EVIDENCE"
        elif (
            any(
                (_text(row.get("primary_action")) or "").upper() == "TRY"
                for row in members
            )
            and not score_ids
        ):
            status = "DATA_GAP"
            exclusion_reason = "UNLINKED_TRY"

        first = members[0]
        final = members[-1]
        semantic_resolved = status == "FINAL"
        semantic_rows = (
            [row for row, _ in semantic_pairs] if semantic_resolved else []
        )
        semantic_tags = (
            sorted(
                {
                    tag
                    for _, tags in semantic_pairs
                    for tag in tags
                }
            )
            if semantic_resolved
            else []
        )
        final_turnover = (
            semantic_resolved
            and any(bool(tags & _TURNOVER_TAGS) for _, tags in semantic_pairs)
        )
        final_pre_home = _integer(first.get("pre_home_score"))
        final_pre_away = _integer(first.get("pre_away_score"))
        final_state = semantic_rows[-1] if semantic_rows else final
        final_post_home = _integer(final_state.get("post_home_score"))
        final_post_away = _integer(final_state.get("post_away_score"))
        final_score_points = 0
        if (
            semantic_resolved
            and final_pre_home is not None
            and final_pre_away is not None
            and final_post_home is not None
            and final_post_away is not None
        ):
            final_score_points = max(
                0, final_post_home - final_pre_home
            ) + max(
                0, final_post_away - final_pre_away
            )
        if not semantic_resolved:
            final_post_home = final_pre_home
            final_post_away = final_pre_away
            final_score_points = 0
            final_turnover = False

        primary_action = (_text(first.get("primary_action")) or "UNKNOWN").upper()
        if status == "NULLIFIED":
            episode_type = "NULLIFIED"
        elif status == "UNRESOLVED_ADJUDICATION_CHAIN":
            episode_type = "UNRESOLVED_ADJUDICATION_CHAIN"
        elif status == "DATA_GAP":
            episode_type = "DATA_GAP"
        elif has_touchdown:
            episode_type = "TOUCHDOWN_CHAIN"
        elif final_score_points and final_turnover:
            episode_type = "SCORE_TURNOVER"
        else:
            episode_type = primary_action

        first_seen_start = _text(first.get("source_interval_start"))
        first_seen_end = _text(first.get("source_interval_end"))
        final_known_start = _text(final.get("source_interval_start"))
        final_known_end = _text(final.get("source_interval_end"))
        finalization_gap_seconds = _anchor_gap_seconds(
            first_seen_end, final_known_end
        )
        if (
            finalization_gap_seconds is not None
            and finalization_gap_seconds < 0
        ):
            status = "DATA_GAP"
            episode_type = "DATA_GAP"
            exclusion_reason = "FINALIZED_ANCHOR_PRECEDES_PROVISIONAL"
            semantic_rows = []
            semantic_tags = []
            final_post_home = final_pre_home
            final_post_away = final_pre_away
            final_score_points = 0
            final_turnover = False
        stage_a_eligible = (
            status == "FINAL"
            and any(
                _boolean(row.get("final_sports_outcome_eligible"))
                for row in semantic_rows
            )
        )
        residual_diagnostic_eligible = (
            status == "FINAL"
            and final_known_end is not None
            and _boolean(final.get("stage_b_information_event_eligible"))
            and (_text(final.get("information_status")) or "").upper() == "FINAL"
        )
        if (
            stage_a_eligible
            and not residual_diagnostic_eligible
            and exclusion_reason is None
        ):
            exclusion_reason = "MISSING_OR_INELIGIBLE_FINAL_KNOWN_INTERVAL"

        episode_material = {
            "schema": SCHEMA_VERSION,
            "game_id": game_id,
            "event_ids": [str(row["event_id"]) for row in members],
            "score_sequence_id": sorted(score_ids),
            "adjudication_sequence_id": sorted(adjudication_ids),
        }
        episode_id = (
            f"{game_id}:finalized:"
            f"{_canonical_sha(episode_material).removeprefix('sha256:')}"
        )
        beneficiary = next(
            (
                _text(row.get("beneficiary_team"))
                for row in semantic_rows
                if _text(row.get("beneficiary_team")) is not None
            ),
            None,
        )
        episode = {
            "schema_version": SCHEMA_VERSION,
            "claim_boundary": CLAIM_BOUNDARY,
            "game_id": game_id,
            "episode_id": episode_id,
            "episode_status": status,
            "episode_type": episode_type,
            "constituent_count": len(members),
            "score_sequence_id": next(iter(score_ids)) if len(score_ids) == 1 else None,
            "adjudication_sequence_id": (
                next(iter(adjudication_ids))
                if len(adjudication_ids) == 1
                else None
            ),
            "quarter": _integer(first.get("quarter")),
            "first_seen_interval_start": first_seen_start,
            "first_seen_interval_end": first_seen_end,
            "final_known_interval_start": final_known_start,
            "final_known_interval_end": final_known_end,
            "finalized_anchor": final_known_end,
            "finalized_anchor_semantics": (
                "SOURCE_INTERVAL_END_OF_LAST_CONSTITUENT_DECIDING_FINAL_SEMANTICS"
            ),
            "finalization_gap_seconds": finalization_gap_seconds,
            "stage_a_eligible": stage_a_eligible,
            "residual_diagnostic_eligible": (
                residual_diagnostic_eligible
            ),
            "exclusion_reason": exclusion_reason,
            "home_team": _text(first.get("home_team")),
            "away_team": _text(first.get("away_team")),
            "final_beneficiary_team": beneficiary,
            "final_pre_home_score": final_pre_home,
            "final_pre_away_score": final_pre_away,
            "final_post_home_score": final_post_home,
            "final_post_away_score": final_post_away,
            "final_score_points": final_score_points,
            "final_turnover": final_turnover,
            "final_outcome_tags": json.dumps(
                semantic_tags, separators=(",", ":")
            ),
            "source_hashes": _source_hashes(members),
        }
        episodes.append(episode)
        for row, tags in zip(members, member_tags, strict=True):
            role = _constituent_role(row, tags)
            information_anchor = _text(row.get("source_interval_end"))
            support_status, support_reason = _provisional_feature_support(
                row,
                tags,
                provisional_anchor=information_anchor,
            )
            provisional_snapshot = {
                "schema": "NFLProvisionalFeatureSnapshotV1",
                "game_id": game_id,
                "provisional_known_at": _text(row.get("known_at")),
                "information_anchor": information_anchor,
                "provisional_primary_event_id": str(row["event_id"]),
                "provisional_primary_action": (
                    _text(row.get("primary_action")) or "UNKNOWN"
                ).upper(),
                "provisional_outcome_tags": sorted(tags),
                "provisional_actor_team": _text(row.get("actor_team")),
                "provisional_beneficiary_team": _text(
                    row.get("beneficiary_team")
                ),
                "provisional_score_points_observed": (
                    _observed_score_points(row)
                ),
                "provisional_turnover_observed": bool(
                    tags & _TURNOVER_TAGS
                ),
                "provisional_yards_gained": _number(
                    row.get("yards_gained")
                ),
                "provisional_return_yards": _number(
                    row.get("return_yards")
                ),
                "provisional_feature_support_status": support_status,
                "provisional_feature_support_reason": support_reason,
            }
            anchor_material = {
                "schema": INFORMATION_ANCHOR_SCHEMA,
                "game_id": game_id,
                "event_id": str(row["event_id"]),
                "order_sequence": _integer(row.get("order_sequence")),
            }
            information_anchor_id = (
                f"{game_id}:information-anchor:"
                f"{_canonical_sha(anchor_material).removeprefix('sha256:')}"
            )
            information_anchors.append(
                {
                    "schema_version": INFORMATION_ANCHOR_SCHEMA,
                    "claim_boundary": CLAIM_BOUNDARY,
                    "game_id": game_id,
                    "information_anchor_id": information_anchor_id,
                    "episode_id": episode_id,
                    "event_id": str(row["event_id"]),
                    "raw_play_id": _text(row.get("raw_play_id")),
                    "order_sequence": _integer(row.get("order_sequence")),
                    "constituent_role": role,
                    "score_sequence_id": _text(
                        row.get("score_sequence_id")
                    ),
                    "adjudication_sequence_id": _text(
                        row.get("adjudication_sequence_id")
                    ),
                    "source_interval_start": _text(
                        row.get("source_interval_start")
                    ),
                    "source_interval_end": information_anchor,
                    "information_anchor": information_anchor,
                    "information_anchor_semantics": (
                        "SOURCE_INTERVAL_END_OF_THIS_CONSTITUENT_FIRST_SEEN"
                    ),
                    "provisional_known_at": provisional_snapshot[
                        "provisional_known_at"
                    ],
                    "provisional_primary_event_id": provisional_snapshot[
                        "provisional_primary_event_id"
                    ],
                    "provisional_primary_action": provisional_snapshot[
                        "provisional_primary_action"
                    ],
                    "provisional_outcome_tags": json.dumps(
                        provisional_snapshot["provisional_outcome_tags"],
                        separators=(",", ":"),
                    ),
                    "provisional_actor_team": provisional_snapshot[
                        "provisional_actor_team"
                    ],
                    "provisional_beneficiary_team": provisional_snapshot[
                        "provisional_beneficiary_team"
                    ],
                    "provisional_score_points_observed": (
                        provisional_snapshot[
                            "provisional_score_points_observed"
                        ]
                    ),
                    "provisional_turnover_observed": provisional_snapshot[
                        "provisional_turnover_observed"
                    ],
                    "provisional_yards_gained": provisional_snapshot[
                        "provisional_yards_gained"
                    ],
                    "provisional_return_yards": provisional_snapshot[
                        "provisional_return_yards"
                    ],
                    "provisional_feature_support_status": support_status,
                    "provisional_feature_support_reason": support_reason,
                    "provisional_snapshot_sha256": _canonical_sha(
                        provisional_snapshot
                    ),
                    "canonical_stage_b_information_event_eligible": (
                        _boolean(
                            row.get("stage_b_information_event_eligible")
                        )
                    ),
                    "primary_selection_eligible": (
                        support_status == "SUPPORTED"
                        and information_anchor is not None
                        and _boolean(
                            row.get("stage_b_information_event_eligible")
                        )
                    ),
                    "censor_boundary_eligible": (
                        information_anchor is not None
                    ),
                    "source_hashes": _source_hashes([row]),
                    "pbp_source_sha256": _require_sha(
                        row.get("pbp_source_sha256"),
                        field="pbp_source_sha256",
                    ),
                }
            )
            constituents.append(
                {
                    "schema_version": CONSTITUENT_SCHEMA,
                    "claim_boundary": CLAIM_BOUNDARY,
                    "game_id": game_id,
                    "event_id": str(row["event_id"]),
                    "raw_play_id": _text(row.get("raw_play_id")),
                    "order_sequence": _integer(row.get("order_sequence")),
                    "episode_id": episode_id,
                    "mapping_status": "MAPPED",
                    "constituent_role": role,
                    "exclusion_reason": exclusion_reason,
                    "score_sequence_id": _text(row.get("score_sequence_id")),
                    "adjudication_sequence_id": _text(
                        row.get("adjudication_sequence_id")
                    ),
                    "source_interval_start": _text(
                        row.get("source_interval_start")
                    ),
                    "source_interval_end": _text(
                        row.get("source_interval_end")
                    ),
                    "final_outcome_tags": episode["final_outcome_tags"],
                    "final_turnover": final_turnover,
                    "final_score_points": final_score_points,
                    "pbp_source_sha256": _require_sha(
                        row.get("pbp_source_sha256"),
                        field="pbp_source_sha256",
                    ),
                }
            )

    constituent_events = [str(row["event_id"]) for row in constituents]
    duplicate_mappings = len(constituent_events) - len(set(constituent_events))
    source_events = set(frame["event_id"].astype(str))
    mapped_events = set(constituent_events)
    silent = len(source_events - mapped_events)
    mapped_count = sum(
        row["mapping_status"] == "MAPPED" for row in constituents
    )
    excluded_count = sum(
        row["mapping_status"] == "EXCLUDED" for row in constituents
    )
    anchor_event_ids = [
        str(row["event_id"]) for row in information_anchors
    ]
    anchor_ids = [
        str(row["information_anchor_id"]) for row in information_anchors
    ]
    episode_ids = {str(row["episode_id"]) for row in episodes}
    anchors_valid = (
        len(information_anchors) == mapped_count
        and len(anchor_event_ids) == len(set(anchor_event_ids))
        and len(anchor_ids) == len(set(anchor_ids))
        and all(
            str(row["episode_id"]) in episode_ids
            for row in information_anchors
        )
        and all(
            not bool(row["primary_selection_eligible"])
            or row["provisional_feature_support_status"] == "SUPPORTED"
            for row in information_anchors
        )
    )
    gate = (
        "PASS"
        if silent == 0
        and duplicate_mappings == 0
        and len(constituents) == len(frame)
        and anchors_valid
        else "FAIL"
    )
    reconciliation = {
        "schema_version": RECONCILIATION_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "game_id": game_id,
        "raw_row_count": len(frame),
        "mapped_row_count": mapped_count,
        "excluded_row_count": excluded_count,
        "episode_count": len(episodes),
        "information_anchor_count": len(information_anchors),
        "stage_a_eligible_episode_count": sum(
            bool(row["stage_a_eligible"]) for row in episodes
        ),
        "residual_diagnostic_eligible_episode_count": sum(
            bool(row["residual_diagnostic_eligible"]) for row in episodes
        ),
        "primary_selection_eligible_anchor_count": sum(
            bool(row["primary_selection_eligible"])
            for row in information_anchors
        ),
        "censor_boundary_eligible_anchor_count": sum(
            bool(row["censor_boundary_eligible"])
            for row in information_anchors
        ),
        "unresolved_adjudication_episode_count": sum(
            row["episode_status"] == "UNRESOLVED_ADJUDICATION_CHAIN"
            for row in episodes
        ),
        "data_gap_episode_count": sum(
            row["episode_status"] == "DATA_GAP" for row in episodes
        ),
        "nullified_episode_count": sum(
            row["episode_status"] == "NULLIFIED" for row in episodes
        ),
        "ot_excluded_row_count": ot_excluded,
        "administrative_excluded_row_count": admin_excluded,
        "raw_rows_silently_dropped": silent,
        "duplicate_event_mapping_count": duplicate_mappings,
        "publication_gate": gate,
    }
    if gate != "PASS":
        raise FinalizedEpisodeError(
            f"raw-to-episode reconciliation failed: {game_id}"
        )
    return episodes, information_anchors, constituents, reconciliation


def build_finalized_episode_tables(
    canonical_events: pd.DataFrame,
) -> FinalizedEpisodeTables:
    """Build deterministic regulation-only episodes from canonical V4 rows."""

    if not isinstance(canonical_events, pd.DataFrame):
        raise TypeError("canonical_events must be a pandas DataFrame")
    missing = sorted(set(_REQUIRED_EVENT_COLUMNS) - set(canonical_events.columns))
    if missing:
        raise FinalizedEpisodeError(
            f"canonical events missing required columns: {missing}"
        )
    if canonical_events.empty:
        return FinalizedEpisodeTables(
            episodes=_empty_frame(_EPISODE_COLUMNS),
            information_anchors=_empty_frame(_INFORMATION_ANCHOR_COLUMNS),
            constituents=_empty_frame(_CONSTITUENT_COLUMNS),
            reconciliation=_empty_frame(_RECONCILIATION_COLUMNS),
        )
    episodes: list[dict[str, object]] = []
    information_anchors: list[dict[str, object]] = []
    constituents: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for _, game in canonical_events.groupby("game_id", sort=True, dropna=False):
        (
            game_episodes,
            game_information_anchors,
            game_constituents,
            game_audit,
        ) = _build_game(game)
        episodes.extend(game_episodes)
        information_anchors.extend(game_information_anchors)
        constituents.extend(game_constituents)
        audits.append(game_audit)
    episode_frame = pd.DataFrame(episodes, columns=list(_EPISODE_COLUMNS))
    anchor_frame = pd.DataFrame(
        information_anchors, columns=list(_INFORMATION_ANCHOR_COLUMNS)
    )
    constituent_frame = pd.DataFrame(
        constituents, columns=list(_CONSTITUENT_COLUMNS)
    )
    audit_frame = pd.DataFrame(audits, columns=list(_RECONCILIATION_COLUMNS))
    if episode_frame["episode_id"].isna().any() or episode_frame[
        "episode_id"
    ].duplicated().any():
        raise FinalizedEpisodeError(
            "each finalized episode must publish exactly one final row"
        )
    if (
        anchor_frame["information_anchor_id"].isna().any()
        or anchor_frame["information_anchor_id"].duplicated().any()
        or anchor_frame["event_id"].isna().any()
        or anchor_frame["event_id"].duplicated().any()
        or not set(anchor_frame["episode_id"]).issubset(
            set(episode_frame["episode_id"])
        )
    ):
        raise FinalizedEpisodeError(
            "provisional information anchor identity or episode FK failed"
        )
    episode_frame = episode_frame.sort_values(
        ["game_id", "first_seen_interval_end", "episode_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    anchor_frame = anchor_frame.sort_values(
        ["game_id", "order_sequence", "event_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    constituent_frame = constituent_frame.sort_values(
        ["game_id", "order_sequence", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    audit_frame = audit_frame.sort_values("game_id", kind="mergesort").reset_index(
        drop=True
    )
    return FinalizedEpisodeTables(
        episodes=episode_frame,
        information_anchors=anchor_frame,
        constituents=constituent_frame,
        reconciliation=audit_frame,
    )


def _read_verified_events(
    *,
    project_root: Path,
    facts_batch_path: Path,
    expected_batch_sha256: str,
    expected_game_count: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    batch_path = facts_batch_path.resolve()
    if not batch_path.is_file():
        raise FinalizedEpisodeError("facts batch is missing")
    observed_batch_sha = _sha256_file(batch_path)
    if observed_batch_sha != expected_batch_sha256:
        raise FinalizedEpisodeError("facts batch hash mismatch")
    try:
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizedEpisodeError("facts batch is not valid JSON") from exc
    if not isinstance(batch, dict):
        raise FinalizedEpisodeError("facts batch must be a JSON object")
    material = dict(batch)
    declared_batch_sha = material.pop("batch_sha256", None)
    if declared_batch_sha != _canonical_sha(material):
        raise FinalizedEpisodeError("facts batch semantic hash mismatch")
    if (
        batch.get("schema") != _FACTS_BATCH_SCHEMA
        or batch.get("publication_id") != _FACTS_PUBLICATION_ID
        or batch.get("experiment_id") != "X-13"
        or batch.get("cohort") != "development"
        or batch.get("game_count") != expected_game_count
        or batch.get("market_data_read") is not False
        or batch.get("holdout_reaction_accessed") is not False
        or batch.get("publication_gate") != "PASS"
    ):
        raise FinalizedEpisodeError("facts batch contract mismatch")
    entries = batch.get("games")
    if not isinstance(entries, list) or len(entries) != expected_game_count:
        raise FinalizedEpisodeError("facts batch game inventory is incomplete")
    dataset_ancestor = next(
        (
            parent
            for parent in batch_path.parents
            if parent.name == _FACTS_PUBLICATION_ID
        ),
        None,
    )
    if dataset_ancestor is None:
        raise FinalizedEpisodeError("facts batch path is outside V4 dataset")
    descriptor_root = dataset_ancestor.parent.resolve()
    if project_root != descriptor_root and project_root not in descriptor_root.parents:
        # Normal publications live below project_root; custom tests do as well.
        if project_root not in batch_path.parents:
            raise FinalizedEpisodeError("facts batch is outside project_root")

    frames: list[pd.DataFrame] = []
    game_ids: list[str] = []
    child_manifest_hashes: list[str] = []
    object_hashes: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise FinalizedEpisodeError("facts child descriptor is invalid")
        game_id = _text(entry.get("game_id"))
        if game_id is None or game_id in game_ids:
            raise FinalizedEpisodeError("facts game identity is invalid")
        game_ids.append(game_id)
        expected_manifest_sha = _require_sha(
            entry.get("manifest_sha256"), field="manifest_sha256"
        )
        manifest_path = _safe_path(
            descriptor_root,
            str(entry.get("manifest_path", "")),
            field=f"{game_id}.manifest_path",
        )
        if _sha256_file(manifest_path) != expected_manifest_sha:
            raise FinalizedEpisodeError(
                f"facts child manifest hash mismatch: {game_id}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise FinalizedEpisodeError("facts child manifest is invalid")
        child_material = dict(manifest)
        bundle_sha = child_material.pop("bundle_sha256", None)
        if (
            manifest.get("schema") != _FACTS_GAME_SCHEMA
            or manifest.get("publication_id") != _FACTS_PUBLICATION_ID
            or manifest.get("experiment_id") != "X-13"
            or manifest.get("cohort") != "development"
            or manifest.get("game_id") != game_id
            or manifest.get("market_data_read") is not False
            or manifest.get("holdout_reaction_accessed") is not False
            or manifest.get("publication_gate") != "PASS"
            or bundle_sha != entry.get("bundle_sha256")
            or bundle_sha != _canonical_sha(child_material)
        ):
            raise FinalizedEpisodeError(
                f"facts child manifest contract mismatch: {game_id}"
            )
        tables = manifest.get("tables")
        if not isinstance(tables, list):
            raise FinalizedEpisodeError("facts table inventory is missing")
        matches = [
            descriptor
            for descriptor in tables
            if isinstance(descriptor, dict)
            and descriptor.get("name") == "canonical_factor_events"
        ]
        if len(matches) != 1:
            raise FinalizedEpisodeError(
                f"canonical fact table is not unique: {game_id}"
            )
        descriptor = matches[0]
        expected_object_sha = _require_sha(
            descriptor.get("object_sha256"), field="object_sha256"
        )
        object_path = _safe_path(
            descriptor_root,
            str(descriptor.get("object_path", "")),
            field=f"{game_id}.object_path",
        )
        if (
            not object_path.is_file()
            or object_path.stat().st_size != descriptor.get("byte_length")
            or _sha256_file(object_path) != expected_object_sha
        ):
            raise FinalizedEpisodeError(
                f"canonical fact object hash mismatch: {game_id}"
            )
        parquet = pq.ParquetFile(object_path)
        if parquet.metadata.num_rows != descriptor.get("row_count"):
            raise FinalizedEpisodeError(
                f"canonical fact row count mismatch: {game_id}"
            )
        available = set(parquet.schema_arrow.names)
        missing = sorted(set(_REQUIRED_EVENT_COLUMNS) - available)
        if missing:
            raise FinalizedEpisodeError(
                f"canonical fact object missing columns: {game_id}: {missing}"
            )
        frame = parquet.read(columns=list(_REQUIRED_EVENT_COLUMNS)).to_pandas()
        if len(frame) != descriptor.get("row_count"):
            raise FinalizedEpisodeError(
                f"canonical fact decode count mismatch: {game_id}"
            )
        if set(frame["game_id"].dropna().astype(str)) != {game_id}:
            raise FinalizedEpisodeError(
                f"canonical fact game mismatch: {game_id}"
            )
        frames.append(frame)
        child_manifest_hashes.append(expected_manifest_sha)
        object_hashes.append(expected_object_sha)
    if len(set(game_ids)) != expected_game_count:
        raise FinalizedEpisodeError("facts game set is incomplete")
    return (
        pd.concat(frames, ignore_index=True),
        {
            "facts_batch_file_sha256": expected_batch_sha256,
            "facts_batch_sha256": declared_batch_sha,
            "facts_game_manifest_hashes_sha256": _canonical_sha(
                sorted(child_manifest_hashes)
            ),
            "canonical_event_object_hashes_sha256": _canonical_sha(
                sorted(object_hashes)
            ),
        },
    )


def _frame_payload(frame: pd.DataFrame) -> tuple[bytes, str, str]:
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
    )
    payload = sink.getvalue().to_pybytes()
    schema_sha = _sha256_bytes(table.schema.serialize().to_pybytes())
    semantic_sha = _canonical_sha(
        {
            "columns": list(frame.columns),
            "rows": _canonical_records(frame),
        }
    )
    return payload, schema_sha, semantic_sha


def _publish_frame(
    *,
    output_root: Path,
    name: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    payload, schema_sha, semantic_sha = _frame_payload(frame)
    object_sha = _sha256_bytes(payload)
    digest = object_sha.removeprefix("sha256:")
    relative = (
        Path(PUBLICATION_ID)
        / "objects"
        / "sha256"
        / digest[:2]
        / f"{digest}.parquet"
    )
    target = _safe_path(output_root, relative, field=f"{name}.object_path")
    _atomic_publish(target, payload)
    return {
        "name": name,
        "object_path": relative.as_posix(),
        "object_sha256": object_sha,
        "byte_length": len(payload),
        "row_count": len(frame),
        "schema_columns": list(frame.columns),
        "schema_fingerprint": schema_sha,
        "semantic_rows_sha256": semantic_sha,
    }


def publish_finalized_episode_batch(
    *,
    project_root: str | Path,
    facts_batch_path: str | Path = DEFAULT_FACTS_BATCH,
    facts_batch_file_sha256: str = DEFAULT_FACTS_BATCH_FILE_SHA256,
    expected_game_count: int = 153,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> FinalizedEpisodePublication:
    """Verify Exact-153 V4, build episodes, and atomically publish one batch."""

    started = time.perf_counter()
    root = Path(project_root).resolve()
    source_path = Path(facts_batch_path)
    if not source_path.is_absolute():
        source_path = _safe_path(root, source_path, field="facts_batch_path")
    output = Path(output_root)
    if not output.is_absolute():
        output = _safe_path(root, output, field="output_root")
    if expected_game_count <= 0:
        raise FinalizedEpisodeError("expected_game_count must be positive")
    expected_sha = _require_sha(
        facts_batch_file_sha256, field="facts_batch_file_sha256"
    )
    events, source_binding = _read_verified_events(
        project_root=root,
        facts_batch_path=source_path,
        expected_batch_sha256=expected_sha,
        expected_game_count=expected_game_count,
    )
    tables = build_finalized_episode_tables(events)
    support_statuses = tables.information_anchors[
        "provisional_feature_support_status"
    ]
    mapped_row_count = int(
        tables.reconciliation["mapped_row_count"].sum()
    )
    if (
        len(tables.reconciliation) != expected_game_count
        or tables.reconciliation["publication_gate"].ne("PASS").any()
        or int(tables.reconciliation["raw_rows_silently_dropped"].sum()) != 0
        or int(tables.reconciliation["raw_row_count"].sum()) != len(events)
        or len(tables.constituents) != len(events)
        or len(tables.information_anchors) != mapped_row_count
        or support_statuses.isna().any()
        or not support_statuses.isin(
            {"SUPPORTED", "PRIMARY_FEATURE_SUPPORT_UNPROVEN"}
        ).all()
        or tables.information_anchors["provisional_snapshot_sha256"].map(
            lambda value: isinstance(value, str)
            and _SHA_RE.fullmatch(value) is not None
        ).eq(False).any()
        or tables.information_anchors.loc[
            tables.information_anchors[
                "primary_selection_eligible"
            ].eq(True),  # noqa: E712
            "provisional_feature_support_status",
        ].ne("SUPPORTED").any()
    ):
        raise FinalizedEpisodeError("batch reconciliation failed")
    descriptors = [
        _publish_frame(
            output_root=output,
            name=name,
            frame=frame,
        )
        for name, frame in (
            ("finalized_episodes", tables.episodes),
            (
                "provisional_information_anchors",
                tables.information_anchors,
            ),
            ("episode_constituents", tables.constituents),
            ("episode_reconciliation", tables.reconciliation),
        )
    ]
    counts = {
        "game_count": expected_game_count,
        "raw_row_count": len(events),
        "mapped_row_count": mapped_row_count,
        "episode_count": len(tables.episodes),
        "information_anchor_count": len(tables.information_anchors),
        "constituent_row_count": len(tables.constituents),
        "stage_a_eligible_episode_count": int(
            tables.episodes["stage_a_eligible"].sum()
        ),
        "residual_diagnostic_eligible_episode_count": int(
            tables.episodes["residual_diagnostic_eligible"].sum()
        ),
        "primary_selection_eligible_anchor_count": int(
            tables.information_anchors["primary_selection_eligible"].sum()
        ),
        "censor_boundary_eligible_anchor_count": int(
            tables.information_anchors["censor_boundary_eligible"].sum()
        ),
        "provisional_feature_supported_anchor_count": int(
            support_statuses.eq("SUPPORTED").sum()
        ),
        "provisional_feature_support_unproven_anchor_count": int(
            support_statuses.eq("PRIMARY_FEATURE_SUPPORT_UNPROVEN").sum()
        ),
        "provisional_known_at_present_anchor_count": int(
            tables.information_anchors["provisional_known_at"].notna().sum()
        ),
        "unresolved_adjudication_episode_count": int(
            tables.reconciliation[
                "unresolved_adjudication_episode_count"
            ].sum()
        ),
        "data_gap_episode_count": int(
            tables.reconciliation["data_gap_episode_count"].sum()
        ),
        "nullified_episode_count": int(
            tables.reconciliation["nullified_episode_count"].sum()
        ),
        "ot_excluded_row_count": int(
            tables.reconciliation["ot_excluded_row_count"].sum()
        ),
        "administrative_excluded_row_count": int(
            tables.reconciliation["administrative_excluded_row_count"].sum()
        ),
        "raw_rows_silently_dropped": 0,
    }
    material = {
        "schema": MANIFEST_SCHEMA,
        "publication_id": PUBLICATION_ID,
        "experiment_id": EXPERIMENT_ID,
        "builder_version": BUILDER_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_scope": SOURCE_SCOPE,
        "market_data_read": False,
        "holdout_reaction_accessed": False,
        "regulation_only": True,
        "blocked_predecessor": {
            "schema": "FinalizedEpisodeV1",
            "publication_id": "finalized-episodes-regulation-v1",
            "status": "BLOCKED_DIAGNOSTIC",
            "reason": "ADR-0007",
        },
        "anchor_kinds": [
            {
                "analysis_role": "PRIMARY_SELECTION",
                "anchor_kind": "PROVISIONAL_FIRST_SEEN",
                "table": "provisional_information_anchors",
                "column": "information_anchor",
            },
            {
                "analysis_role": "RESIDUAL_CORRECTION_DIAGNOSTIC",
                "anchor_kind": "FINALIZED",
                "table": "finalized_episodes",
                "column": "finalized_anchor",
            },
        ],
        "provisional_feature_contract": {
            "schema": "NFLProvisionalFeatureSnapshotV1",
            "table": "provisional_information_anchors",
            "selection_rule": (
                "EACH_MAPPED_CONSTITUENT_SNAPSHOT_NO_FINAL_OUTCOME_DEPENDENCY"
            ),
            "known_at_column": "provisional_known_at",
            "source_interval_start_column": "source_interval_start",
            "source_interval_end_column": "source_interval_end",
            "anchor_column": "information_anchor",
            "support_status_column": (
                "provisional_feature_support_status"
            ),
            "support_reason_column": (
                "provisional_feature_support_reason"
            ),
            "snapshot_sha256_column": "provisional_snapshot_sha256",
            "source_target_eligibility_column": (
                "canonical_stage_b_information_event_eligible"
            ),
            "primary_eligibility_rule": (
                "SUPPORTED_AND_INFORMATION_ANCHOR_PRESENT_AND_CANONICAL_"
                "STAGE_B_INFORMATION_EVENT_ELIGIBLE"
            ),
            "censor_boundary_rule": (
                "ALL_MAPPED_REGULATION_INFORMATION_CONSTITUENTS_WITH_"
                "SOURCE_INTERVAL_END"
            ),
            "next_censor_time_column": "source_interval_start",
            "overlap_rule": "ORDER_AMBIGUOUS_EXCLUDE",
            "feature_columns": [
                "provisional_primary_event_id",
                "provisional_primary_action",
                "provisional_outcome_tags",
                "provisional_actor_team",
                "provisional_beneficiary_team",
                "provisional_score_points_observed",
                "provisional_turnover_observed",
                "provisional_yards_gained",
                "provisional_return_yards",
            ],
            "forbidden_primary_tables": ["finalized_episodes"],
            "forbidden_primary_columns": [
                "episode_status",
                "episode_type",
                "final_beneficiary_team",
                "final_pre_home_score",
                "final_pre_away_score",
                "final_post_home_score",
                "final_post_away_score",
                "final_score_points",
                "final_turnover",
                "final_outcome_tags",
                "finalized_anchor",
            ],
            "review_revision_policy": (
                "UNPROVEN_WITHOUT_AUDITABLE_REVISION_CHRONOLOGY"
            ),
        },
        "source": source_binding,
        "counts": counts,
        "tables": descriptors,
        "publication_gate": "PASS",
    }
    bundle_sha = _canonical_sha(material)
    manifest = {**material, "bundle_sha256": bundle_sha}
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha = _sha256_bytes(manifest_bytes)
    digest = manifest_sha.removeprefix("sha256:")
    relative = (
        Path(PUBLICATION_ID)
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.manifest.json"
    )
    manifest_path = _safe_path(
        output, relative, field="finalized_episode_manifest"
    )
    _atomic_publish(manifest_path, manifest_bytes)
    if _sha256_file(manifest_path) != manifest_sha:
        raise FinalizedEpisodeError("published manifest hash mismatch")
    for descriptor in descriptors:
        target = _safe_path(
            output,
            str(descriptor["object_path"]),
            field=f"{descriptor['name']}.published_object",
        )
        if (
            target.stat().st_size != descriptor["byte_length"]
            or _sha256_file(target) != descriptor["object_sha256"]
            or pq.ParquetFile(target).metadata.num_rows
            != descriptor["row_count"]
        ):
            raise FinalizedEpisodeError(
                f"published object verification failed: {descriptor['name']}"
            )
    return FinalizedEpisodePublication(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        bundle_sha256=bundle_sha,
        game_count=counts["game_count"],
        raw_row_count=counts["raw_row_count"],
        information_anchor_count=counts["information_anchor_count"],
        episode_count=counts["episode_count"],
        stage_a_eligible_episode_count=counts[
            "stage_a_eligible_episode_count"
        ],
        residual_diagnostic_eligible_episode_count=counts[
            "residual_diagnostic_eligible_episode_count"
        ],
        primary_selection_eligible_anchor_count=counts[
            "primary_selection_eligible_anchor_count"
        ],
        censor_boundary_eligible_anchor_count=counts[
            "censor_boundary_eligible_anchor_count"
        ],
        unresolved_adjudication_episode_count=counts[
            "unresolved_adjudication_episode_count"
        ],
        data_gap_episode_count=counts["data_gap_episode_count"],
        nullified_episode_count=counts["nullified_episode_count"],
        ot_excluded_row_count=counts["ot_excluded_row_count"],
        raw_rows_silently_dropped=counts["raw_rows_silently_dropped"],
        runtime_seconds=time.perf_counter() - started,
        peak_rss_mib=_peak_rss_mib(),
    )


__all__ = [
    "DEFAULT_FACTS_BATCH",
    "DEFAULT_FACTS_BATCH_FILE_SHA256",
    "DEFAULT_OUTPUT_ROOT",
    "FinalizedEpisodeError",
    "FinalizedEpisodePublication",
    "FinalizedEpisodeTables",
    "build_finalized_episode_tables",
    "publish_finalized_episode_batch",
]
