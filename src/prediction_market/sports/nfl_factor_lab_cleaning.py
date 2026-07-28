"""Fail-closed row disposition for NFL Factor Lab sports inputs.

The feature view is intentionally narrower than the source replay.  This
module preserves that distinction explicitly: every selected source row is
represented in an immutable ledger even when the row cannot enter ordinary
play-factor calculations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from numbers import Integral, Real
from typing import Any, Literal


class SportsRowCleaningError(ValueError):
    """Source rows cannot be assigned deterministic sports dispositions."""


SportsRowType = Literal[
    "live_play",
    "administrative",
    "review",
    "no_play",
    "deleted",
    "unknown",
]
SportsRowDisposition = Literal[
    "factor_eligible",
    "administrative_only",
    "audit_only",
]

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_LIVE_PLAY_TYPES = frozenset(
    {
        "pass",
        "run",
        "qb_kneel",
        "qb_spike",
        "sack",
        "punt",
        "field_goal",
        "kickoff",
        "extra_point",
        "two_point_attempt",
        "try",
        "kneel",
        "spike",
    }
)
_LIVE_INDICATORS = (
    "pass_attempt",
    "rush_attempt",
    "sack",
    "qb_kneel",
    "qb_spike",
    "punt_attempt",
    "field_goal_attempt",
    "kickoff_attempt",
    "extra_point_attempt",
    "two_point_attempt",
)


def _records(value: object, *, field: str) -> list[dict[str, Any]]:
    if hasattr(value, "to_pylist"):
        value = value.to_pylist()
    elif hasattr(value, "to_dict") and not isinstance(value, Mapping):
        value = value.to_dict(orient="records")
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise SportsRowCleaningError(f"{field} must be a record sequence")
    result: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise SportsRowCleaningError(f"{field} contains a non-record")
        result.append(dict(row))
    return result


def _stable_id(value: object, *, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise SportsRowCleaningError(f"{field} is missing")
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    if type(value) is str and value and value == value.strip():
        return value
    raise SportsRowCleaningError(f"{field} is not a stable scalar")


def _optional_id(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _stable_id(value, field=field)


def _text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
    if type(value) is not str:
        raise SportsRowCleaningError("text field has a non-string value")
    stripped = value.strip()
    return stripped or None


def _indicator(value: object, *, field: str) -> bool:
    if value is None:
        return False
    if type(value) is bool:
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            return False
        if math.isfinite(number) and number in {0.0, 1.0}:
            return bool(int(number))
    raise SportsRowCleaningError(f"{field} must be binary")


def _semantic_value(value: object) -> object:
    if value is None or type(value) in {str, int, bool}:
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        return None if not math.isfinite(number) else {"$decimal": str(number)}
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise SportsRowCleaningError("semantic datetime must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_semantic_value(child) for child in value]
    raise SportsRowCleaningError(
        f"semantic row contains unsupported {type(value).__name__}"
    )


def semantic_rows_sha256(value: object) -> str:
    """Hash stable semantic values rather than file size or row count."""

    payload = json.dumps(
        _semantic_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _source_time_present(
    raw: Mapping[str, object],
    canonical: Mapping[str, object],
) -> bool:
    return any(
        value is not None and _text(value) is not None
        for value in (
            canonical.get("source_time_start_utc"),
            canonical.get("source_time_end_utc"),
            raw.get("source_time_start_utc"),
            raw.get("source_time_end_utc"),
            raw.get("time_of_day"),
        )
    )


def _normalized_description(
    raw: Mapping[str, object],
    canonical: Mapping[str, object],
) -> str:
    description = _text(
        canonical.get(
            "description",
            raw.get("desc", raw.get("description")),
        )
    )
    return "" if description is None else " ".join(description.upper().split())


def _administrative_marker(description: str) -> bool:
    return bool(
        description == "GAME"
        or description.startswith("END ")
        or description.startswith("OT END ")
        or description in {"END", "GAME END"}
    )


def _live_action(
    raw: Mapping[str, object],
    canonical: Mapping[str, object],
) -> bool:
    play_family = _text(canonical.get("play_family"))
    play_type = _text(raw.get("play_type"))
    if play_family is not None:
        return play_family in _LIVE_PLAY_TYPES
    if play_type is not None and play_type.lower() in _LIVE_PLAY_TYPES:
        return True
    return any(_indicator(raw.get(field), field=field) for field in _LIVE_INDICATORS)


def _review(
    raw: Mapping[str, object],
    canonical: Mapping[str, object],
) -> tuple[bool, bool]:
    review = (
        _indicator(raw.get("replay_or_challenge"), field="replay_or_challenge")
        or _indicator(raw.get("review"), field="review")
        or bool(canonical.get("review"))
    )
    result = (
        _text(canonical.get("review_result"))
        or _text(raw.get("replay_or_challenge_result"))
        or _text(raw.get("review_result"))
        or ""
    ).lower()
    reversed_result = "reversed" in result
    return review or bool(result and result != "none"), reversed_result


def _episode_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for source in rows:
        episode = dict(source)
        game_id = _stable_id(episode.get("game_id"), field="episode game_id")
        play_ids = episode.get("play_ids")
        if not isinstance(play_ids, Sequence) or isinstance(
            play_ids, (str, bytes, bytearray)
        ):
            raise SportsRowCleaningError("episode play_ids must be an array")
        for value in play_ids:
            key = (
                game_id,
                _stable_id(value, field="episode play_id"),
            )
            if key in result:
                raise SportsRowCleaningError(
                    "one canonical play is assigned to multiple episodes"
                )
            result[key] = episode
    return result


def _canonical_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for source in rows:
        canonical = dict(source)
        game_id = _stable_id(
            canonical.get("game_id"),
            field="canonical game_id",
        )
        raw_play_id = _stable_id(
            canonical.get("raw_play_id", canonical.get("play_id")),
            field="canonical raw_play_id",
        )
        key = (game_id, raw_play_id)
        if key in result:
            raise SportsRowCleaningError("canonical rows contain a duplicate raw key")
        result[key] = canonical
    return result


@dataclass(frozen=True, slots=True)
class SportsRowDispositionV2:
    """One raw/current-game row and its authoritative sports disposition."""

    game_id: str
    raw_play_id: str
    canonical_play_id: str | None
    episode_id: str | None
    row_type: SportsRowType
    live_play_outcome_eligible: bool
    administrative_event_eligible: bool
    factor_eligible: bool
    disposition: SportsRowDisposition
    exclusion_reasons: tuple[str, ...]
    source_hash: str

    def __post_init__(self) -> None:
        _stable_id(self.game_id, field="game_id")
        _stable_id(self.raw_play_id, field="raw_play_id")
        _optional_id(self.canonical_play_id, field="canonical_play_id")
        _optional_id(self.episode_id, field="episode_id")
        if self.row_type not in {
            "live_play",
            "administrative",
            "review",
            "no_play",
            "deleted",
            "unknown",
        }:
            raise SportsRowCleaningError("row_type is unsupported")
        for value, field in (
            (self.live_play_outcome_eligible, "live_play_outcome_eligible"),
            (self.administrative_event_eligible, "administrative_event_eligible"),
            (self.factor_eligible, "factor_eligible"),
        ):
            if type(value) is not bool:
                raise SportsRowCleaningError(f"{field} must be boolean")
        if self.factor_eligible and not self.live_play_outcome_eligible:
            raise SportsRowCleaningError(
                "factor eligibility requires a live-play outcome"
            )
        expected_disposition: SportsRowDisposition = (
            "factor_eligible"
            if self.factor_eligible
            else (
                "administrative_only"
                if self.administrative_event_eligible
                else "audit_only"
            )
        )
        if self.disposition != expected_disposition:
            raise SportsRowCleaningError("disposition disagrees with eligibility")
        if self.exclusion_reasons != tuple(sorted(set(self.exclusion_reasons))):
            raise SportsRowCleaningError(
                "exclusion_reasons must be sorted and unique"
            )
        if not _SHA256_RE.fullmatch(self.source_hash):
            raise SportsRowCleaningError("source_hash must be sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "game_id": self.game_id,
            "raw_play_id": self.raw_play_id,
            "canonical_play_id": self.canonical_play_id,
            "episode_id": self.episode_id,
            "row_type": self.row_type,
            "live_play_outcome_eligible": self.live_play_outcome_eligible,
            "administrative_event_eligible": self.administrative_event_eligible,
            "factor_eligible": self.factor_eligible,
            "disposition": self.disposition,
            "exclusion_reasons": list(self.exclusion_reasons),
            "source_hash": self.source_hash,
        }


@dataclass(frozen=True, slots=True)
class SportsRowDispositionLedgerV2:
    """Canonical semantic ledger for every selected raw/current-game row."""

    rows: tuple[SportsRowDispositionV2, ...]
    source_hashes: tuple[tuple[str, str], ...]
    rows_sha256: str
    schema_version: str = "SportsRowDispositionLedgerV2"

    def __post_init__(self) -> None:
        if self.schema_version != "SportsRowDispositionLedgerV2":
            raise SportsRowCleaningError("ledger schema version changed")
        if not self.rows:
            raise SportsRowCleaningError("ledger rows must not be empty")
        if self.source_hashes != tuple(sorted(self.source_hashes)):
            raise SportsRowCleaningError("source_hashes must use canonical order")
        names = [name for name, _digest in self.source_hashes]
        if len(names) != len(set(names)):
            raise SportsRowCleaningError("source hash names must be unique")
        for name, digest in self.source_hashes:
            _stable_id(name, field="source hash name")
            if not _SHA256_RE.fullmatch(digest):
                raise SportsRowCleaningError("source hash must be sha256")
        if self.rows_sha256 != self._expected_rows_sha256(
            self.rows,
            self.source_hashes,
        ):
            raise SportsRowCleaningError("rows_sha256 does not match ledger rows")

    @staticmethod
    def _expected_rows_sha256(
        rows: Sequence[SportsRowDispositionV2],
        source_hashes: Sequence[tuple[str, str]],
    ) -> str:
        return semantic_rows_sha256(
            {
                "schema_version": "SportsRowDispositionLedgerV2",
                "source_hashes": [
                    {"name": name, "sha256": digest}
                    for name, digest in source_hashes
                ],
                "rows": [row.to_dict() for row in rows],
            }
        )

    @classmethod
    def build(
        cls,
        *,
        rows: Sequence[SportsRowDispositionV2],
        source_hashes: Mapping[str, str],
    ) -> SportsRowDispositionLedgerV2:
        canonical_sources = tuple(
            sorted((str(name), str(digest)) for name, digest in source_hashes.items())
        )
        canonical_rows = tuple(rows)
        return cls(
            rows=canonical_rows,
            source_hashes=canonical_sources,
            rows_sha256=cls._expected_rows_sha256(
                canonical_rows,
                canonical_sources,
            ),
        )

    def to_records(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]


def build_sports_row_disposition_ledger(
    *,
    selected_pbp_rows: Sequence[Mapping[str, object]],
    finalized_episode_rows: Sequence[Mapping[str, object]],
    source_hashes: Mapping[str, str],
    canonical_event_rows: Sequence[Mapping[str, object]] = (),
) -> SportsRowDispositionLedgerV2:
    """Preserve and classify every selected source row without silent drops."""

    selected = _records(selected_pbp_rows, field="selected_pbp_rows")
    if not selected:
        raise SportsRowCleaningError("selected_pbp_rows must not be empty")
    episodes = _episode_index(
        _records(finalized_episode_rows, field="finalized_episode_rows")
    )
    canonical = _canonical_index(
        _records(canonical_event_rows, field="canonical_event_rows")
    )
    indexed: list[tuple[tuple[str, int, int, str], SportsRowDispositionV2]] = []
    for input_ordinal, raw in enumerate(selected):
        game_id = _stable_id(raw.get("game_id"), field="game_id")
        raw_play_id = _stable_id(raw.get("play_id"), field="play_id")
        canonical_row = canonical.get((game_id, raw_play_id), {})
        canonical_play_id = _stable_id(
            canonical_row.get(
                "canonical_play_id",
                canonical_row.get("play_id", raw_play_id),
            ),
            field="canonical_play_id",
        )
        episode = episodes.get((game_id, canonical_play_id))
        episode_id = (
            None
            if episode is None
            else _stable_id(episode.get("episode_id"), field="episode_id")
        )
        play_type = (_text(raw.get("play_type")) or "").lower()
        no_play = (
            play_type == "no_play"
            or bool(canonical_row.get("no_play"))
            or _indicator(raw.get("no_play"), field="no_play")
        )
        deleted = (
            bool(canonical_row.get("deleted"))
            or _indicator(raw.get("play_deleted"), field="play_deleted")
            or _indicator(raw.get("deleted"), field="deleted")
        )
        review, reversed_result = _review(raw, canonical_row)
        audit_only = bool(episode and episode.get("audit_only"))
        nullified = bool(episode and episode.get("nullified"))
        description = _normalized_description(raw, canonical_row)
        live_action = _live_action(raw, canonical_row)
        explicit_admin = bool(
            _administrative_marker(description)
            or bool(canonical_row.get("game_end"))
            or _indicator(raw.get("game_end"), field="game_end")
            or _indicator(raw.get("timeout"), field="timeout")
            or (
                _indicator(raw.get("penalty"), field="penalty")
                and not live_action
            )
        )
        administrative = no_play or nullified or review or explicit_admin
        if deleted:
            row_type: SportsRowType = "deleted"
        elif no_play:
            row_type = "no_play"
        elif review:
            row_type = "review"
        elif explicit_admin or (nullified and not live_action):
            row_type = "administrative"
        elif live_action:
            row_type = "live_play"
        else:
            row_type = "unknown"
        live_play_outcome_eligible = bool(
            live_action
            and not no_play
            and not nullified
            and not deleted
            and not review
            and not audit_only
            and episode is not None
        )
        administrative_event_eligible = bool(
            administrative and not deleted and not audit_only and episode is not None
        )
        source_time_present = _source_time_present(raw, canonical_row)
        factor_eligible = live_play_outcome_eligible and source_time_present
        reasons: set[str] = set()
        if no_play:
            reasons.add("no_play")
        if nullified:
            reasons.add("episode_nullified")
        if deleted:
            reasons.add("deleted")
        if reversed_result:
            reasons.add("reversed")
        if review:
            reasons.add("review")
        if administrative:
            reasons.add("administrative")
        if not source_time_present:
            reasons.add("source_time_missing")
        if audit_only:
            reasons.add("episode_audit_only")
        if episode is None:
            reasons.add("missing_episode")
        if row_type == "unknown":
            reasons.add("unclassified_row")
        disposition: SportsRowDisposition = (
            "factor_eligible"
            if factor_eligible
            else (
                "administrative_only"
                if administrative_event_eligible
                else "audit_only"
            )
        )
        order_value = raw.get("order_sequence", raw.get("play_id"))
        if isinstance(order_value, bool) or not isinstance(order_value, Real):
            raise SportsRowCleaningError("order_sequence must be numeric")
        order_number = float(order_value)
        if not math.isfinite(order_number) or not order_number.is_integer():
            raise SportsRowCleaningError("order_sequence must be an integer")
        indexed.append(
            (
                (game_id, int(order_number), input_ordinal, raw_play_id),
                SportsRowDispositionV2(
                    game_id=game_id,
                    raw_play_id=raw_play_id,
                    canonical_play_id=canonical_play_id,
                    episode_id=episode_id,
                    row_type=row_type,
                    live_play_outcome_eligible=live_play_outcome_eligible,
                    administrative_event_eligible=administrative_event_eligible,
                    factor_eligible=factor_eligible,
                    disposition=disposition,
                    exclusion_reasons=tuple(sorted(reasons)),
                    source_hash=semantic_rows_sha256(
                        {
                            "schema_version": "SportsRawSourceRowV2",
                            "row": raw,
                        }
                    ),
                ),
            )
        )
    return SportsRowDispositionLedgerV2.build(
        rows=[row for _key, row in sorted(indexed, key=lambda item: item[0])],
        source_hashes=source_hashes,
    )


__all__ = [
    "SportsRowCleaningError",
    "SportsRowDispositionLedgerV2",
    "SportsRowDispositionV2",
    "build_sports_row_disposition_ledger",
    "semantic_rows_sha256",
]
