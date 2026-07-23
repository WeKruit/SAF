"""Independent field census for the frozen 2025 nflverse play-by-play season.

The reducer adapter constructs the actual transition.  The oracle in this
module parses native rows separately and derives its own expected post-state;
it deliberately does not call any reducer row/snapshot helper.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

from prediction_market.contracts import canonical_json_bytes
from prediction_market.sports import nfl_game_state as nfl
from prediction_market.sports.x11 import (
    X11_DATASET_ID,
    X11_FROZEN_PARTITION_ALLOWLIST,
    X11_LICENSE_REF,
    X11_LICENSE_STATUS,
    X11_NFLVERSE_VERSION,
    expected_nflverse_source_cursor,
)
from prediction_market.static_store import (
    StaticStoreError,
    read_verified_static_object,
)


_DATASET_ID = X11_DATASET_ID
_SOURCE_VERSION = X11_NFLVERSE_VERSION
_RULEBOOK_VERSION = "2025"
_HASH_PREFIX = "sha256:"

_CENSUS_COLUMNS = (
    "game_id",
    "play_id",
    "order_sequence",
    "season_type",
    "qtr",
    "quarter_seconds_remaining",
    "game_seconds_remaining",
    "time",
    "home_team",
    "away_team",
    "fixed_drive",
    "goal_to_go",
    "play_clock",
    "posteam",
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
    "quarter_end",
    "desc",
    "first_down",
    "interception",
    "fumble_lost",
    "timeout",
    "timeout_team",
)

_AUDIT_BASES: dict[
    str,
    Literal["native_direct", "native_derived", "rule_derived", "lineage"],
] = {
    "game_id": "native_direct",
    "home_team": "native_direct",
    "away_team": "native_direct",
    "season_type": "native_direct",
    "source_play_id": "native_direct",
    "source_order_sequence": "native_direct",
    "period": "native_direct",
    "period_seconds_remaining": "native_direct",
    "game_seconds_remaining": "native_direct",
    "terminal": "native_direct",
    "home_score": "native_derived",
    "away_score": "native_derived",
    "possession_team": "native_derived",
    "down": "native_derived",
    "distance": "native_derived",
    "yardline_100": "native_derived",
    "drive_id": "native_derived",
    "play_clock_seconds": "native_derived",
    "goal_to_go": "native_derived",
    "timeout_observed": "native_derived",
    "timeout_observed_team": "native_derived",
    "timeout_kind": "native_derived",
    "home_timeouts_remaining": "rule_derived",
    "away_timeouts_remaining": "rule_derived",
    "timeout_charge_team": "rule_derived",
    "suspended": "rule_derived",
    "lifecycle_action": "rule_derived",
    "carry_forward_context": "rule_derived",
    "clock_correction": "rule_derived",
    "clock_correction_observed_period_seconds_remaining": "native_direct",
    "clock_correction_observed_game_seconds_remaining": "native_direct",
    "quarter_end": "rule_derived",
    "postseason_third_timeout_zero": "rule_derived",
    "context_source_play_id": "lineage",
    "context_source_order_sequence": "lineage",
    "event_context_source_play_id": "lineage",
    "event_context_source_order_sequence": "lineage",
    "source_window_play_ids": "lineage",
    "source_window_order_sequences": "lineage",
    "sequence": "lineage",
    "last_event_id": "lineage",
}


class NFLSeasonCensusError(ValueError):
    """The input cannot support a governed NFL season census."""


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
class NFLSeasonCensusFailure:
    game_id: str
    source_order_sequence: int | None
    field: str
    expected: object | None
    actual: object | None
    error: str


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
    failures: tuple[NFLSeasonCensusFailure, ...]

    def to_document(self) -> dict[str, object]:
        document = asdict(self)
        document["lifecycle_counts"] = dict(self.lifecycle_counts)
        document["quality_flag_counts"] = dict(self.quality_flag_counts)
        document["reducer_latency"] = dict(self.reducer_latency)
        return document


@dataclass(frozen=True, slots=True)
class _OracleTransition:
    state: Mapping[str, object]
    lifecycle_action: str
    timeout_observed: bool
    timeout_observed_team: str | None
    timeout_kind: str
    timeout_charge_team: str | None
    carry_forward_context: bool
    clock_correction: bool
    clock_correction_observed_period_seconds_remaining: int | None
    clock_correction_observed_game_seconds_remaining: int | None
    quarter_end: bool
    postseason_third_timeout_zero: bool
    context_source_play_id: str
    context_source_order_sequence: int
    source_window_play_ids: tuple[str, ...]
    source_window_order_sequences: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ScanResult:
    completed_games: int
    fail_closed_games: int
    transitions: int
    field_audits: tuple[NFLFieldAudit, ...]
    lifecycle_counts: tuple[tuple[str, int], ...]
    quality_flag_counts: tuple[tuple[str, int], ...]
    canonical_state_sha256: str
    failures: tuple[NFLSeasonCensusFailure, ...]


class _OracleViolation(ValueError):
    def __init__(
        self,
        *,
        field: str,
        expected: object | None,
        actual: object | None,
        message: str,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.expected = expected
        self.actual = actual


class _FieldMismatch(_OracleViolation):
    pass


def _unwrap(value: object) -> object:
    as_py = getattr(value, "as_py", None)
    return as_py() if callable(as_py) else value


def _missing(value: object) -> bool:
    value = _unwrap(value)
    if value is None:
        return True
    if isinstance(value, Decimal):
        return not value.is_finite()
    if isinstance(value, Real) and not isinstance(value, Integral):
        return not math.isfinite(float(value))
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _value(
    row: Mapping[str, object],
    field: str,
    *,
    required: bool,
) -> object | None:
    if field not in row or _missing(row.get(field)):
        if required:
            raise _OracleViolation(
                field=field,
                expected="nonmissing native value",
                actual=None,
                message=f"native row requires nonmissing {field}",
            )
        return None
    return _unwrap(row[field])


def _int_value(
    row: Mapping[str, object],
    field: str,
    *,
    required: bool = True,
) -> int | None:
    value = _value(row, field, required=required)
    if value is None:
        return None
    if isinstance(value, bool):
        raise _OracleViolation(
            field=field,
            expected="integer",
            actual=value,
            message=f"native {field} must be an integer",
        )
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Decimal) and value == value.to_integral_value():
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
    raise _OracleViolation(
        field=field,
        expected="integer",
        actual=value,
        message=f"native {field} must be an integer",
    )


def _text_value(
    row: Mapping[str, object],
    field: str,
    *,
    required: bool = True,
) -> str | None:
    value = _value(row, field, required=required)
    if value is None:
        return None
    if type(value) is not str or not value or value != value.strip():
        raise _OracleViolation(
            field=field,
            expected="canonical nonempty string",
            actual=value,
            message=f"native {field} must be canonical text",
        )
    return value


def _indicator(
    row: Mapping[str, object],
    field: str,
    *,
    default: bool,
) -> bool:
    value = _value(row, field, required=False)
    if value is None:
        return default
    if type(value) is bool:
        return value
    if isinstance(value, Integral) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, Real) and float(value) in {0.0, 1.0}:
        return bool(int(float(value)))
    raise _OracleViolation(
        field=field,
        expected="binary observation",
        actual=value,
        message=f"native {field} must be binary",
    )


def _source_play_id(row: Mapping[str, object]) -> str:
    value = _value(row, "play_id", required=True)
    assert value is not None
    if isinstance(value, bool):
        raise _OracleViolation(
            field="play_id",
            expected="stable scalar",
            actual=value,
            message="native play_id must be a stable scalar",
        )
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    if type(value) is str and value and value == value.strip():
        return value
    raise _OracleViolation(
        field="play_id",
        expected="stable scalar",
        actual=value,
        message="native play_id must be a stable scalar",
    )


def _stable_optional(
    row: Mapping[str, object],
    field: str,
) -> str | None:
    value = _value(row, field, required=False)
    if value is None:
        return None
    if isinstance(value, bool):
        raise _OracleViolation(
            field=field,
            expected="stable scalar",
            actual=value,
            message=f"native {field} must be a stable scalar",
        )
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    if type(value) is str and value and value == value.strip():
        return value
    raise _OracleViolation(
        field=field,
        expected="stable scalar",
        actual=value,
        message=f"native {field} must be a stable scalar",
    )


def _play_clock(row: Mapping[str, object]) -> int | None:
    value = _value(row, "play_clock", required=False)
    if value is None:
        return None
    if type(value) is str and value.isdigit():
        number = int(value)
    elif isinstance(value, Integral) and not isinstance(value, bool):
        number = int(value)
    elif isinstance(value, Real) and float(value).is_integer():
        number = int(value)
    else:
        raise _OracleViolation(
            field="play_clock",
            expected="integer seconds",
            actual=value,
            message="native play_clock must be integer seconds",
        )
    if not 0 <= number <= 40:
        raise _OracleViolation(
            field="play_clock",
            expected="[0, 40]",
            actual=number,
            message="native play_clock is outside [0, 40]",
        )
    return number


def _quarter_seconds(row: Mapping[str, object]) -> int:
    observed = _int_value(
        row,
        "quarter_seconds_remaining",
        required=False,
    )
    if observed is not None:
        return observed
    clock = _text_value(row, "time")
    assert clock is not None
    parts = clock.split(":")
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise _OracleViolation(
            field="time",
            expected="MM:SS",
            actual=clock,
            message="native time must be MM:SS",
        )
    minutes, seconds = (int(part) for part in parts)
    if minutes > 15 or seconds > 59:
        raise _OracleViolation(
            field="time",
            expected="valid period clock",
            actual=clock,
            message="native time is outside a valid period clock",
        )
    return minutes * 60 + seconds


def _canonical_game_id(row: Mapping[str, object]) -> str:
    native = _text_value(row, "game_id")
    assert native is not None
    return native if native.startswith("game_nflverse_") else f"game_nflverse_{native}"


def _lifecycle_action(row: Mapping[str, object]) -> str:
    if _text_value(row, "play_type_nfl", required=False) != "COMMENT":
        return "none"
    description = _text_value(row, "desc", required=False)
    if description is not None and description.startswith(
        "The game has been suspended."
    ):
        return "suspend"
    if description is not None and description.startswith(
        "The game has resumed."
    ):
        return "resume"
    return "none"


def _timeout_kind(row: Mapping[str, object]) -> str:
    observed = _indicator(row, "timeout", default=False)
    administrative = (
        _text_value(row, "play_type_nfl", required=False) == "TIMEOUT"
        and _text_value(row, "play_type", required=False) == "no_play"
    )
    if administrative and not observed:
        raise _OracleViolation(
            field="timeout",
            expected=True,
            actual=False,
            message="TIMEOUT + no_play source must charge a timeout",
        )
    if not observed:
        return "none"
    return "administrative" if administrative else "play_attached"


def _quarter_end(row: Mapping[str, object]) -> bool:
    observed = _indicator(row, "quarter_end", default=False)
    native_type = (
        _text_value(row, "play_type_nfl", required=False)
        == "END_QUARTER"
    )
    if observed != native_type:
        raise _OracleViolation(
            field="quarter_end",
            expected=native_type,
            actual=observed,
            message="native quarter_end disagrees with play_type_nfl",
        )
    return observed


def _normalize_timeout(
    native_value: int,
    *,
    season_type: str,
    period: int,
) -> int:
    offset = 1 if season_type == "POST" and period >= 5 else 0
    normalized = native_value + offset
    maximum = 2 if season_type == "REG" and period >= 5 else 3
    if not 0 <= normalized <= maximum:
        raise _OracleViolation(
            field="timeout_counter",
            expected=f"integer in [0, {maximum}] after rule normalization",
            actual=normalized,
            message="native timeout counter is outside the rules allotment",
        )
    return normalized


def _mapped_scores(
    row: Mapping[str, object],
    *,
    after_play: bool,
) -> tuple[int, int]:
    home_team = _text_value(row, "home_team")
    away_team = _text_value(row, "away_team")
    posteam = _text_value(row, "posteam", required=False)
    assert home_team is not None
    assert away_team is not None
    suffix = "_post" if after_play else ""
    if posteam is None:
        posteam_value = _value(
            row,
            f"posteam_score{suffix}",
            required=False,
        )
        defteam_value = _value(
            row,
            f"defteam_score{suffix}",
            required=False,
        )
        if posteam_value is not None or defteam_value is not None:
            raise _OracleViolation(
                field="score_source_consistency",
                expected="no team-relative score without posteam",
                actual=(posteam_value, defteam_value),
                message="no-possession row carries team-relative score",
            )
        home = _int_value(row, "total_home_score")
        away = _int_value(row, "total_away_score")
        assert home is not None
        assert away is not None
        return home, away

    if posteam not in {home_team, away_team}:
        raise _OracleViolation(
            field="posteam",
            expected=(home_team, away_team),
            actual=posteam,
            message="posteam is not a game participant",
        )
    offense = _int_value(row, f"posteam_score{suffix}")
    defense = _int_value(row, f"defteam_score{suffix}")
    assert offense is not None
    assert defense is not None
    mapped = (
        (offense, defense) if posteam == home_team else (defense, offense)
    )
    if after_play:
        totals = (
            _int_value(row, "total_home_score"),
            _int_value(row, "total_away_score"),
        )
        if mapped != totals:
            raise _OracleViolation(
                field="score_source_consistency",
                expected=totals,
                actual=mapped,
                message="team-relative post-play score disagrees with totals",
            )
    return mapped


def _native_snapshot(
    row: Mapping[str, object],
) -> dict[str, object]:
    game_id = _canonical_game_id(row)
    season_type = _text_value(row, "season_type")
    home_team = _text_value(row, "home_team")
    away_team = _text_value(row, "away_team")
    assert season_type is not None
    assert home_team is not None
    assert away_team is not None
    if season_type not in {"REG", "POST"}:
        raise _OracleViolation(
            field="season_type",
            expected=("REG", "POST"),
            actual=season_type,
            message="native season_type must be REG or POST",
        )
    period = _int_value(row, "qtr")
    assert period is not None
    lifecycle = _lifecycle_action(row)
    if lifecycle == "none":
        period_seconds = _quarter_seconds(row)
        game_seconds = _int_value(row, "game_seconds_remaining")
        assert game_seconds is not None
    else:
        period_seconds = None
        game_seconds = None
    posteam = _text_value(row, "posteam", required=False)
    down = _int_value(row, "down", required=False)
    if posteam is None and down is not None:
        raise _OracleViolation(
            field="context_carry_source",
            expected="down absent when posteam is absent",
            actual=down,
            message="context-carry row contains down without possession",
        )
    distance = (
        _int_value(row, "ydstogo") if down is not None else None
    )
    yardline = _int_value(row, "yardline_100", required=False)
    if down is not None and yardline is None:
        raise _OracleViolation(
            field="yardline_100",
            expected="present when down is present",
            actual=None,
            message="native down requires yardline_100",
        )
    home_score, away_score = _mapped_scores(row, after_play=False)
    native_home_timeouts = _int_value(row, "home_timeouts_remaining")
    native_away_timeouts = _int_value(row, "away_timeouts_remaining")
    assert native_home_timeouts is not None
    assert native_away_timeouts is not None
    return {
        "game_id": game_id,
        "season_type": season_type,
        "home_team": home_team,
        "away_team": away_team,
        "period": period,
        "period_seconds_remaining": period_seconds,
        "game_seconds_remaining": game_seconds,
        "source_play_id": _source_play_id(row),
        "source_order_sequence": _int_value(row, "order_sequence"),
        "drive_id": _stable_optional(row, "fixed_drive"),
        "play_clock_seconds": _play_clock(row),
        "possession_team": posteam,
        "down": down,
        "distance": distance,
        "yardline_100": yardline,
        "goal_to_go": _indicator(row, "goal_to_go", default=False),
        "home_score": home_score,
        "away_score": away_score,
        "home_timeouts_remaining": _normalize_timeout(
            native_home_timeouts,
            season_type=season_type,
            period=period,
        ),
        "away_timeouts_remaining": _normalize_timeout(
            native_away_timeouts,
            season_type=season_type,
            period=period,
        ),
    }


def _infer_terminal(row: Mapping[str, object]) -> bool:
    description = _text_value(row, "desc", required=False)
    return description is not None and description.upper() in {
        "END GAME",
        "END OF GAME",
    }


def _lineage_event_id(
    *,
    raw_object_sha256: str,
    source_version: str,
    game_id: str,
    source_order_sequence: int,
    next_source_order_sequence: int,
    source_window_play_ids: tuple[str, ...],
    source_window_order_sequences: tuple[int, ...],
    source_window_raw_record_ordinals: tuple[int, ...],
) -> str:
    """Bind the source row and every ordered offline successor member."""

    material = {
        "dataset_id": _DATASET_ID,
        "raw_object_sha256": raw_object_sha256,
        "source_version": source_version,
        "game_id": game_id,
        "source_order_sequence": source_order_sequence,
        "next_source_order_sequence": next_source_order_sequence,
        "source_window_play_ids": source_window_play_ids,
        "source_window_order_sequences": source_window_order_sequences,
        "source_window_raw_record_ordinals": (
            source_window_raw_record_ordinals
        ),
    }
    digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    return f"evt_{digest}"


def _clock_correction_expected(
    prior_state: Mapping[str, object],
    pre_row: Mapping[str, object],
    post_row: Mapping[str, object],
) -> bool:
    if _timeout_kind(pre_row) != "administrative":
        return False
    source_play_id = _source_play_id(pre_row)
    next_play_id = _source_play_id(post_row)
    if not source_play_id.isdigit() or not next_play_id.isdigit():
        return False
    if int(source_play_id) <= int(next_play_id):
        return False
    period = _int_value(pre_row, "qtr")
    assert period is not None
    if period != prior_state["period"]:
        return False
    period_seconds = _quarter_seconds(pre_row)
    game_seconds = _int_value(pre_row, "game_seconds_remaining")
    assert game_seconds is not None
    return (
        period_seconds > int(prior_state["period_seconds_remaining"])
        or game_seconds > int(prior_state["game_seconds_remaining"])
    )


def _oracle_initial_state(
    row: Mapping[str, object],
) -> dict[str, object]:
    """Derive initial expectations directly from one native source row."""

    snapshot = _native_snapshot(row)
    if snapshot["period_seconds_remaining"] is None:
        raise _OracleViolation(
            field="clock",
            expected="complete initial clock",
            actual=None,
            message="initial native row cannot be a lifecycle row",
        )
    return {
        "sport": "nfl",
        **snapshot,
        "context_source_play_id": snapshot["source_play_id"],
        "context_source_order_sequence": snapshot[
            "source_order_sequence"
        ],
        "sequence": 0,
        "terminal": False,
        "suspended": False,
        "last_event_id": None,
    }


def _timeout_charge_team(
    prior_state: Mapping[str, object],
    *,
    target_period: int,
    target_home_timeouts: int,
    target_away_timeouts: int,
) -> str | None:
    """Derive a charged team from counters, independently of timeout flags."""

    home_delta = (
        target_home_timeouts
        - int(prior_state["home_timeouts_remaining"])
    )
    away_delta = (
        target_away_timeouts
        - int(prior_state["away_timeouts_remaining"])
    )
    if home_delta == -1 and away_delta == 0:
        return str(prior_state["home_team"])
    if away_delta == -1 and home_delta == 0:
        return str(prior_state["away_team"])
    if home_delta == 0 and away_delta == 0:
        return None

    period_changed = target_period == int(prior_state["period"]) + 1
    season_type = str(prior_state["season_type"])
    reset = (
        period_changed
        and (
            (
                int(prior_state["period"]) == 2
                and target_period == 3
            )
            or (
                season_type == "REG"
                and int(prior_state["period"]) == 4
                and target_period == 5
            )
            or (
                season_type == "POST"
                and target_period >= 5
                and target_period % 2 == 1
            )
        )
    )
    reset_allotment = (
        2 if season_type == "REG" and target_period >= 5 else 3
    )
    if (
        reset
        and target_home_timeouts == reset_allotment
        and target_away_timeouts == reset_allotment
    ):
        return None
    raise _OracleViolation(
        field="timeout_charge_team",
        expected="one-team decrement, no change, or a rules reset",
        actual=(home_delta, away_delta),
        message="timeout counter transition is not a legal charge/reset",
    )


def _timeout_reset_allotment(
    prior_state: Mapping[str, object],
    *,
    target_period: int,
) -> int | None:
    previous_period = int(prior_state["period"])
    if target_period != previous_period + 1:
        return None
    if (previous_period, target_period) == (2, 3):
        return 3
    season_type = str(prior_state["season_type"])
    if season_type == "REG":
        return 2 if (previous_period, target_period) == (4, 5) else None
    if target_period >= 5 and target_period % 2 == 1:
        return 3
    return None


def _oracle_transition(
    prior_state: Mapping[str, object],
    pre_row: Mapping[str, object],
    post_row: Mapping[str, object],
    *,
    successor_rows: tuple[Mapping[str, object], ...],
    sequence: int,
    event_id: str,
) -> _OracleTransition:
    """Derive one expected transition without reducer adapter helpers."""

    pre = _native_snapshot(pre_row)
    post = _native_snapshot(post_row)
    for field in (
        "game_id",
        "season_type",
        "home_team",
        "away_team",
    ):
        if pre[field] != post[field]:
            raise _OracleViolation(
                field=field,
                expected=pre[field],
                actual=post[field],
                message=f"native pre/post rows disagree on {field}",
            )
        if prior_state[field] != pre[field]:
            raise _OracleViolation(
                field=field,
                expected=prior_state[field],
                actual=pre[field],
                message=f"prior state and native pre row disagree on {field}",
            )
    for field in ("source_play_id", "source_order_sequence"):
        if prior_state[field] != pre[field]:
            raise _OracleViolation(
                field=field,
                expected=prior_state[field],
                actual=pre[field],
                message=f"prior state and native pre row disagree on {field}",
            )
    if int(post["source_order_sequence"]) <= int(
        pre["source_order_sequence"]
    ):
        raise _OracleViolation(
            field="order_sequence",
            expected=f">{pre['source_order_sequence']}",
            actual=post["source_order_sequence"],
            message="native source order must strictly increase",
        )

    lifecycle_action = _lifecycle_action(pre_row)
    post_lifecycle_action = _lifecycle_action(post_row)
    timeout_kind = _timeout_kind(pre_row)
    post_timeout_kind = _timeout_kind(post_row)
    source_carries_context = (
        lifecycle_action != "none"
        or timeout_kind == "administrative"
        or _quarter_end(pre_row)
    )
    if not successor_rows or successor_rows[0] is not post_row:
        raise _OracleViolation(
            field="source_window_order_sequences",
            expected="window beginning at immediate successor",
            actual=tuple(
                _validated_order(row) for row in successor_rows
            ),
            message="oracle successor window does not begin at post_row",
        )
    context_row = (
        successor_rows[-1]
        if _text_value(
            successor_rows[-1],
            "posteam",
            required=False,
        )
        is not None
        else None
    )
    context_snapshot = (
        None if context_row is None else _native_snapshot(context_row)
    )
    if context_snapshot is not None:
        for field in (
            "game_id",
            "season_type",
            "home_team",
            "away_team",
        ):
            if context_snapshot[field] != pre[field]:
                raise _OracleViolation(
                    field=field,
                    expected=pre[field],
                    actual=context_snapshot[field],
                    message=f"lookahead context disagrees on {field}",
                )
    carry_forward_context = (
        source_carries_context
        or (
            post["possession_team"] is None
            and context_snapshot is None
        )
    )
    context = (
        prior_state
        if carry_forward_context
        else post
        if post["possession_team"] is not None
        else context_snapshot
    )
    assert context is not None

    clock_correction = _clock_correction_expected(
        prior_state,
        pre_row,
        post_row,
    )
    if lifecycle_action != "none":
        target_period = prior_state["period"]
        target_period_seconds = prior_state["period_seconds_remaining"]
        target_game_seconds = prior_state["game_seconds_remaining"]
    elif timeout_kind == "administrative":
        target_period = pre["period"]
        target_period_seconds = (
            prior_state["period_seconds_remaining"]
            if clock_correction
            else pre["period_seconds_remaining"]
        )
        target_game_seconds = (
            prior_state["game_seconds_remaining"]
            if clock_correction
            else pre["game_seconds_remaining"]
        )
    elif post_lifecycle_action != "none" or (
        post_timeout_kind == "administrative"
        and post["period"] == pre["period"]
    ):
        target_period = pre["period"]
        target_period_seconds = pre["period_seconds_remaining"]
        target_game_seconds = pre["game_seconds_remaining"]
    else:
        target_period = post["period"]
        target_period_seconds = post["period_seconds_remaining"]
        target_game_seconds = post["game_seconds_remaining"]

    home_score, away_score = _mapped_scores(pre_row, after_play=True)
    observed_timeout_team = (
        _text_value(pre_row, "timeout_team")
        if timeout_kind != "none"
        else None
    )
    if lifecycle_action == "suspend":
        if bool(prior_state["suspended"]):
            raise _OracleViolation(
                field="suspended",
                expected=False,
                actual=True,
                message="second suspension is invalid",
            )
        suspended = True
    elif lifecycle_action == "resume":
        if not bool(prior_state["suspended"]):
            raise _OracleViolation(
                field="suspended",
                expected=True,
                actual=False,
                message="resume without suspension is invalid",
            )
        suspended = False
    else:
        suspended = bool(prior_state["suspended"])

    reset_allotment = _timeout_reset_allotment(
        prior_state,
        target_period=int(target_period),
    )
    if timeout_kind != "none":
        target_home_timeouts = int(
            pre["home_timeouts_remaining"]
        )
        target_away_timeouts = int(
            pre["away_timeouts_remaining"]
        )
    elif reset_allotment is not None:
        target_home_timeouts = reset_allotment
        target_away_timeouts = reset_allotment
    else:
        target_home_timeouts = int(
            prior_state["home_timeouts_remaining"]
        )
        target_away_timeouts = int(
            prior_state["away_timeouts_remaining"]
        )
    if (
        timeout_kind == "none"
        and post_timeout_kind == "none"
        and (
            int(post["home_timeouts_remaining"])
            != target_home_timeouts
            or int(post["away_timeouts_remaining"])
            != target_away_timeouts
        )
    ):
        raise _OracleViolation(
            field="timeout_counter",
            expected=(
                target_home_timeouts,
                target_away_timeouts,
            ),
            actual=(
                post["home_timeouts_remaining"],
                post["away_timeouts_remaining"],
            ),
            message="native successor timeout counters changed without cause",
        )
    timeout_charge_team = _timeout_charge_team(
        prior_state,
        target_period=int(target_period),
        target_home_timeouts=target_home_timeouts,
        target_away_timeouts=target_away_timeouts,
    )
    if timeout_charge_team is not None and (
        timeout_kind == "none"
        or observed_timeout_team != timeout_charge_team
    ):
        raise _OracleViolation(
            field="timeout_charge_team",
            expected=observed_timeout_team,
            actual=timeout_charge_team,
            message="counter charge does not match the observed timeout team",
        )
    target_timeout = (
        target_home_timeouts
        if observed_timeout_team == pre["home_team"]
        else target_away_timeouts
    )
    native_timeout = (
        _int_value(
            pre_row,
            "home_timeouts_remaining"
            if observed_timeout_team == pre["home_team"]
            else "away_timeouts_remaining",
        )
        if observed_timeout_team is not None
        else None
    )
    postseason_third_timeout_zero = (
        pre["season_type"] == "POST"
        and int(pre["period"]) >= 5
        and timeout_kind == "administrative"
        and native_timeout == -1
        and target_timeout == 0
    )

    expected_state = {
        "sport": "nfl",
        "game_id": prior_state["game_id"],
        "sequence": sequence,
        "terminal": (
            False
            if lifecycle_action != "none"
            else _infer_terminal(post_row)
        ),
        "season_type": prior_state["season_type"],
        "home_team": prior_state["home_team"],
        "away_team": prior_state["away_team"],
        "period": target_period,
        "period_seconds_remaining": target_period_seconds,
        "game_seconds_remaining": target_game_seconds,
        "source_play_id": post["source_play_id"],
        "source_order_sequence": post["source_order_sequence"],
        "context_source_play_id": (
            prior_state["context_source_play_id"]
            if carry_forward_context
            else context["source_play_id"]
        ),
        "context_source_order_sequence": (
            prior_state["context_source_order_sequence"]
            if carry_forward_context
            else context["source_order_sequence"]
        ),
        "suspended": suspended,
        "drive_id": context["drive_id"],
        "play_clock_seconds": context["play_clock_seconds"],
        "possession_team": context["possession_team"],
        "down": context["down"],
        "distance": context["distance"],
        "yardline_100": context["yardline_100"],
        "goal_to_go": context["goal_to_go"],
        "home_score": home_score,
        "away_score": away_score,
        "home_timeouts_remaining": target_home_timeouts,
        "away_timeouts_remaining": target_away_timeouts,
        "last_event_id": event_id,
    }
    expected_context_play_id = str(
        expected_state["context_source_play_id"]
    )
    expected_context_order = int(
        expected_state["context_source_order_sequence"]
    )
    source_window = (pre_row, *successor_rows)
    window_play_ids = tuple(
        _source_play_id(row) for row in source_window
    )
    window_orders = tuple(
        _validated_order(row) for row in source_window
    )
    return _OracleTransition(
        state=expected_state,
        lifecycle_action=lifecycle_action,
        timeout_observed=timeout_kind != "none",
        timeout_observed_team=observed_timeout_team,
        timeout_kind=timeout_kind,
        timeout_charge_team=timeout_charge_team,
        carry_forward_context=carry_forward_context,
        clock_correction=clock_correction,
        clock_correction_observed_period_seconds_remaining=(
            int(pre["period_seconds_remaining"])
            if clock_correction
            else None
        ),
        clock_correction_observed_game_seconds_remaining=(
            int(pre["game_seconds_remaining"])
            if clock_correction
            else None
        ),
        quarter_end=_quarter_end(pre_row),
        postseason_third_timeout_zero=postseason_third_timeout_zero,
        context_source_play_id=expected_context_play_id,
        context_source_order_sequence=expected_context_order,
        source_window_play_ids=window_play_ids,
        source_window_order_sequences=window_orders,
    )


def _actual_event(
    state: nfl.NFLGameState,
    pre_row: Mapping[str, object],
    post_row: Mapping[str, object],
    *,
    successor_rows: tuple[Mapping[str, object], ...],
    sequence: int,
    raw_object_sha256: str,
    source_version: str,
) -> nfl.NFLPlayEvent:
    payload = nfl.nflverse_transition_payload(
        state,
        pre_row,
        successor_rows,
        sequence=sequence,
    )
    payload["quality_flags"] = tuple(payload["quality_flags"])
    payload["source_window_play_ids"] = tuple(
        payload["source_window_play_ids"]
    )
    payload["source_window_order_sequences"] = tuple(
        payload["source_window_order_sequences"]
    )
    source_window = (pre_row, *successor_rows)
    event_id = _lineage_event_id(
        raw_object_sha256=raw_object_sha256,
        source_version=source_version,
        game_id=str(payload["game_id"]),
        source_order_sequence=int(payload["source_order_sequence"]),
        next_source_order_sequence=int(
            payload["next_source_order_sequence"]
        ),
        source_window_play_ids=tuple(
            _source_play_id(row) for row in source_window
        ),
        source_window_order_sequences=tuple(
            _validated_order(row) for row in source_window
        ),
        source_window_raw_record_ordinals=tuple(
            _raw_record_ordinal(row) for row in source_window
        ),
    )
    return nfl.NFLPlayEvent(event_id=event_id, **payload)


class _Auditor:
    def __init__(self) -> None:
        self._counts: dict[str, list[int]] = {
            field: [0, 0, 0] for field in _AUDIT_BASES
        }

    def compare(
        self,
        field: str,
        expected: object,
        actual: object,
    ) -> None:
        counts = self._counts[field]
        counts[0] += 1
        if expected == actual:
            counts[1] += 1
            return
        counts[2] += 1
        raise _FieldMismatch(
            field=field,
            expected=expected,
            actual=actual,
            message=f"{field} mismatch: expected {expected!r}, got {actual!r}",
        )

    def reports(self) -> tuple[NFLFieldAudit, ...]:
        return tuple(
            NFLFieldAudit(
                field=field,
                basis=basis,
                comparisons=self._counts[field][0],
                matches=self._counts[field][1],
                mismatches=self._counts[field][2],
            )
            for field, basis in sorted(_AUDIT_BASES.items())
        )


_STATE_AUDIT_FIELDS = tuple(
    field
    for field in _AUDIT_BASES
    if field
    not in {
        "timeout_observed",
        "timeout_observed_team",
        "timeout_kind",
        "timeout_charge_team",
        "lifecycle_action",
        "carry_forward_context",
        "clock_correction",
        "clock_correction_observed_period_seconds_remaining",
        "clock_correction_observed_game_seconds_remaining",
        "quarter_end",
        "postseason_third_timeout_zero",
        "event_context_source_play_id",
        "event_context_source_order_sequence",
        "source_window_play_ids",
        "source_window_order_sequences",
    }
)


def _audit_state(
    auditor: _Auditor,
    expected: Mapping[str, object],
    actual: nfl.NFLGameState,
) -> None:
    for field in _STATE_AUDIT_FIELDS:
        auditor.compare(field, expected[field], getattr(actual, field))


def _failure_field(
    error: Exception,
    *,
    oracle: _OracleTransition | None,
    pre_row: Mapping[str, object],
    stage: str,
) -> str:
    message = str(error)
    if (
        "context-carry event cannot claim" in message
        and _indicator(pre_row, "first_down", default=False)
    ):
        return "first_down"
    for field in sorted(_AUDIT_BASES, key=len, reverse=True):
        if field in message:
            return field
    if oracle is not None and "possession" in message:
        return "possession_team"
    return stage


def _failure(
    *,
    game_id: str,
    order: int | None,
    field: str,
    error: Exception,
    expected: object | None = None,
    actual: object | None = None,
) -> NFLSeasonCensusFailure:
    return NFLSeasonCensusFailure(
        game_id=game_id,
        source_order_sequence=order,
        field=field,
        expected=expected,
        actual=actual,
        error=str(error),
    )


def _state_hash_update(
    digest: Any,
    *,
    game_id: str,
    state: nfl.NFLGameState,
) -> None:
    """Hash each canonical actual state in sorted game/source order."""

    digest.update(
        canonical_json_bytes(
            {
                "game_id": game_id,
                "state": asdict(state),
            }
        )
    )
    digest.update(b"\n")


def _failure_hash_update(
    digest: Any,
    failure: NFLSeasonCensusFailure,
) -> None:
    digest.update(canonical_json_bytes({"failure": asdict(failure)}))
    digest.update(b"\n")


def _validated_order(
    row: Mapping[str, object],
) -> int:
    value = _int_value(row, "order_sequence")
    assert value is not None
    if value < 0:
        raise _OracleViolation(
            field="order_sequence",
            expected="nonnegative integer",
            actual=value,
            message="native order_sequence must be nonnegative",
        )
    return value


def _raw_record_ordinal(row: Mapping[str, object]) -> int:
    value = _int_value(row, "_raw_record_ordinal")
    assert value is not None
    if value < 0:
        raise _OracleViolation(
            field="raw_record_ordinal",
            expected="nonnegative integer",
            actual=value,
            message="raw record ordinal must be nonnegative",
        )
    return value


def _causal_successor_window(
    ordered_rows: Sequence[Mapping[str, object]],
    *,
    source_row: Mapping[str, object],
    post_index: int,
) -> tuple[Mapping[str, object], ...]:
    """Return every row through the first complete causal successor.

    The skipped rows remain explicit raw lineage.  This is an offline
    adjacent-observation aid, not a claim that the context was live-PIT at the
    source event.
    """

    if (
        _timeout_kind(source_row) == "administrative"
        or _lifecycle_action(source_row) != "none"
        or _quarter_end(source_row)
    ):
        return (ordered_rows[post_index],)

    window: list[Mapping[str, object]] = []
    for candidate in ordered_rows[post_index:]:
        window.append(candidate)
        if _text_value(candidate, "posteam", required=False) is not None:
            break
    if not window:
        raise _OracleViolation(
            field="source_window_order_sequences",
            expected="at least the immediate successor",
            actual=(),
            message="successor window is empty",
        )
    return tuple(window)


def _scan_once(
    *,
    grouped_rows: Mapping[str, tuple[Mapping[str, object], ...]],
    raw_object_sha256: str,
    source_version: str,
    latency_samples: list[int],
) -> _ScanResult:
    auditor = _Auditor()
    completed_games = 0
    transitions = 0
    failures: list[NFLSeasonCensusFailure] = []
    lifecycle_counts: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    digest = hashlib.sha256()

    for native_game_id in sorted(grouped_rows):
        game_rows = grouped_rows[native_game_id]
        failed_order: int | None = None
        try:
            ordered = sorted(game_rows, key=_validated_order)
            orders = tuple(_validated_order(row) for row in ordered)
            duplicates = tuple(
                order
                for previous, order in zip(orders, orders[1:])
                if order <= previous
            )
            if duplicates:
                failed_order = duplicates[0]
                raise _OracleViolation(
                    field="order_sequence",
                    expected="strictly increasing unique values",
                    actual=duplicates[0],
                    message=(
                        f"game {native_game_id} has duplicate/non-increasing "
                        f"order_sequence {duplicates[0]}"
                    ),
                )
            if len(ordered) < 2:
                raise _OracleViolation(
                    field="rows",
                    expected="at least two rows",
                    actual=len(ordered),
                    message=f"game {native_game_id} has fewer than two rows",
                )
            failed_order = orders[0]
            oracle_state = _oracle_initial_state(ordered[0])
            actual_state = nfl.state_from_nflverse_row(ordered[0])
            _audit_state(auditor, oracle_state, actual_state)
            _state_hash_update(
                digest,
                game_id=native_game_id,
                state=actual_state,
            )

            for sequence, (pre_row, post_row) in enumerate(
                zip(ordered, ordered[1:]),
                start=1,
            ):
                failed_order = _validated_order(pre_row)
                successor_window = _causal_successor_window(
                    ordered,
                    source_row=pre_row,
                    post_index=sequence,
                )
                lineage_window = (pre_row, *successor_window)
                event_id = _lineage_event_id(
                    raw_object_sha256=raw_object_sha256,
                    source_version=source_version,
                    game_id=str(oracle_state["game_id"]),
                    source_order_sequence=failed_order,
                    next_source_order_sequence=_validated_order(post_row),
                    source_window_play_ids=tuple(
                        _source_play_id(row)
                        for row in lineage_window
                    ),
                    source_window_order_sequences=tuple(
                        _validated_order(row)
                        for row in lineage_window
                    ),
                    source_window_raw_record_ordinals=tuple(
                        _raw_record_ordinal(row)
                        for row in lineage_window
                    ),
                )
                oracle_transition = _oracle_transition(
                    oracle_state,
                    pre_row,
                    post_row,
                    successor_rows=successor_window,
                    sequence=sequence,
                    event_id=event_id,
                )
                try:
                    event = _actual_event(
                        actual_state,
                        pre_row,
                        post_row,
                        successor_rows=successor_window,
                        sequence=sequence,
                        raw_object_sha256=raw_object_sha256,
                        source_version=source_version,
                    )
                except (nfl.NFLGameStateError, TypeError, ValueError) as error:
                    field = _failure_field(
                        error,
                        oracle=oracle_transition,
                        pre_row=pre_row,
                        stage="event_adapter",
                    )
                    raise _OracleViolation(
                        field=field,
                        expected=(
                            getattr(oracle_transition, field)
                            if hasattr(oracle_transition, field)
                            else None
                        ),
                        actual=None,
                        message=f"event adapter failed: {error}",
                    ) from error

                auditor.compare(
                    "lifecycle_action",
                    oracle_transition.lifecycle_action,
                    event.lifecycle_action,
                )
                auditor.compare(
                    "timeout_observed",
                    oracle_transition.timeout_observed,
                    event.timeout_observed,
                )
                auditor.compare(
                    "timeout_observed_team",
                    oracle_transition.timeout_observed_team,
                    event.timeout_observed_team,
                )
                auditor.compare(
                    "timeout_kind",
                    oracle_transition.timeout_kind,
                    event.timeout_kind,
                )
                auditor.compare(
                    "timeout_charge_team",
                    oracle_transition.timeout_charge_team,
                    event.timeout_charge_team,
                )
                auditor.compare(
                    "carry_forward_context",
                    oracle_transition.carry_forward_context,
                    event.carry_forward_context,
                )
                auditor.compare(
                    "clock_correction",
                    oracle_transition.clock_correction,
                    event.clock_correction,
                )
                auditor.compare(
                    "clock_correction_observed_period_seconds_remaining",
                    (
                        oracle_transition
                        .clock_correction_observed_period_seconds_remaining
                    ),
                    (
                        event
                        .clock_correction_observed_period_seconds_remaining
                    ),
                )
                auditor.compare(
                    "clock_correction_observed_game_seconds_remaining",
                    (
                        oracle_transition
                        .clock_correction_observed_game_seconds_remaining
                    ),
                    (
                        event
                        .clock_correction_observed_game_seconds_remaining
                    ),
                )
                auditor.compare(
                    "quarter_end",
                    oracle_transition.quarter_end,
                    event.quarter_end,
                )
                auditor.compare(
                    "event_context_source_play_id",
                    oracle_transition.context_source_play_id,
                    event.context_source_play_id,
                )
                auditor.compare(
                    "event_context_source_order_sequence",
                    oracle_transition.context_source_order_sequence,
                    event.context_source_order_sequence,
                )
                auditor.compare(
                    "source_window_play_ids",
                    oracle_transition.source_window_play_ids,
                    event.source_window_play_ids,
                )
                auditor.compare(
                    "source_window_order_sequences",
                    oracle_transition.source_window_order_sequences,
                    event.source_window_order_sequences,
                )
                if oracle_transition.postseason_third_timeout_zero:
                    expected_zero = 0
                    actual_zero = (
                        event.home_timeouts_remaining
                        if event.timeout_charge_team
                        == actual_state.home_team
                        else event.away_timeouts_remaining
                    )
                    auditor.compare(
                        "postseason_third_timeout_zero",
                        expected_zero,
                        actual_zero,
                    )

                start_ns = time.perf_counter_ns()
                try:
                    # This is the only timed region: no row normalization,
                    # oracle work, envelope material, I/O, or model inference.
                    next_state = nfl.NFL_GAME_STATE_REDUCER.reduce(
                        actual_state,
                        event,
                    )
                except (nfl.NFLGameStateError, TypeError, ValueError) as error:
                    elapsed_ns = time.perf_counter_ns() - start_ns
                    latency_samples.append(max(1, elapsed_ns))
                    field = _failure_field(
                        error,
                        oracle=oracle_transition,
                        pre_row=pre_row,
                        stage="reducer",
                    )
                    raise _OracleViolation(
                        field=field,
                        expected=oracle_transition.state.get(field),
                        actual=None,
                        message=f"reducer failed: {error}",
                    ) from error
                latency_samples.append(
                    max(1, time.perf_counter_ns() - start_ns)
                )
                _audit_state(
                    auditor,
                    oracle_transition.state,
                    next_state,
                )
                if event.lifecycle_action != "none":
                    lifecycle_counts[event.lifecycle_action] += 1
                quality_flag_counts.update(event.quality_flags)
                transitions += 1
                oracle_state = dict(oracle_transition.state)
                actual_state = next_state
                _state_hash_update(
                    digest,
                    game_id=native_game_id,
                    state=actual_state,
                )
        except _OracleViolation as error:
            failure = _failure(
                game_id=native_game_id,
                order=failed_order,
                field=error.field,
                expected=error.expected,
                actual=error.actual,
                error=error,
            )
            failures.append(failure)
            _failure_hash_update(digest, failure)
            continue
        except (nfl.NFLGameStateError, TypeError, ValueError) as error:
            failure = _failure(
                game_id=native_game_id,
                order=failed_order,
                field=_failure_field(
                    error,
                    oracle=None,
                    pre_row=game_rows[0],
                    stage="initial_state",
                ),
                error=error,
            )
            failures.append(failure)
            _failure_hash_update(digest, failure)
            continue
        completed_games += 1

    return _ScanResult(
        completed_games=completed_games,
        fail_closed_games=len(failures),
        transitions=transitions,
        field_audits=auditor.reports(),
        lifecycle_counts=tuple(sorted(lifecycle_counts.items())),
        quality_flag_counts=tuple(sorted(quality_flag_counts.items())),
        canonical_state_sha256=f"{_HASH_PREFIX}{digest.hexdigest()}",
        failures=tuple(failures),
    )


def _canonical_digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value.startswith(_HASH_PREFIX)
        or len(value) != len(_HASH_PREFIX) + 64
    ):
        raise NFLSeasonCensusError(f"{field} must be a canonical SHA-256 digest")
    try:
        int(value.removeprefix(_HASH_PREFIX), 16)
    except ValueError as error:
        raise NFLSeasonCensusError(
            f"{field} must be a canonical SHA-256 digest"
        ) from error
    return value


def _latency_report(samples: Sequence[int]) -> dict[str, int]:
    """Summarize reducer-call nanoseconds across every complete scan run."""

    if not samples:
        return {
            "samples": 0,
            "p50_ns": 0,
            "p95_ns": 0,
            "p99_ns": 0,
            "max_ns": 0,
            "mean_ns": 0,
            "operations_per_second": 0,
        }
    ordered = sorted(samples)

    def percentile(probability: float) -> int:
        index = max(0, math.ceil(probability * len(ordered)) - 1)
        return int(ordered[index])

    mean_ns = max(1, sum(ordered) // len(ordered))
    return {
        "samples": len(ordered),
        "p50_ns": percentile(0.50),
        "p95_ns": percentile(0.95),
        "p99_ns": percentile(0.99),
        "max_ns": int(ordered[-1]),
        "mean_ns": mean_ns,
        "operations_per_second": int(1_000_000_000 // mean_ns),
    }


def census_loaded_nflverse_season(
    *,
    rows: Sequence[Mapping[str, object]],
    raw_object_sha256: str,
    source_version: str,
    scan_runs: int = 2,
) -> NFLSeasonCensusReport:
    """Run repeated field-level audits over already loaded native rows."""

    _canonical_digest(raw_object_sha256, "raw_object_sha256")
    if type(source_version) is not str or not source_version.strip():
        raise NFLSeasonCensusError(
            "source_version must be a canonical nonempty string"
        )
    if type(scan_runs) is not int or scan_runs < 2:
        raise NFLSeasonCensusError("scan_runs must be an integer >= 2")
    if (
        isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or not rows
    ):
        raise NFLSeasonCensusError("rows must be a nonempty sequence")

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise NFLSeasonCensusError(
                f"row {ordinal} must be a mapping-like native row"
            )
        copied = dict(row)
        copied["_raw_record_ordinal"] = ordinal
        try:
            native_game_id = _text_value(copied, "game_id")
        except _OracleViolation as error:
            raise NFLSeasonCensusError(
                f"row {ordinal} has invalid game_id: {error}"
            ) from error
        assert native_game_id is not None
        grouped[native_game_id].append(copied)
    frozen_groups = {
        game_id: tuple(game_rows)
        for game_id, game_rows in grouped.items()
    }

    latency_samples: list[int] = []
    summaries = tuple(
        _scan_once(
            grouped_rows=frozen_groups,
            raw_object_sha256=raw_object_sha256,
            source_version=source_version,
            latency_samples=latency_samples,
        )
        for _ in range(scan_runs)
    )
    first = summaries[0]
    deterministic = all(
        (
            summary.canonical_state_sha256,
            summary.completed_games,
            summary.fail_closed_games,
            summary.transitions,
            summary.field_audits,
            summary.lifecycle_counts,
            summary.quality_flag_counts,
            summary.failures,
        )
        == (
            first.canonical_state_sha256,
            first.completed_games,
            first.fail_closed_games,
            first.transitions,
            first.field_audits,
            first.lifecycle_counts,
            first.quality_flag_counts,
            first.failures,
        )
        for summary in summaries[1:]
    )
    return NFLSeasonCensusReport(
        census_version="v0",
        reducer_version=nfl.NFL_GAME_STATE_REDUCER.reducer_version,
        dataset_id=_DATASET_ID,
        source_version=source_version,
        rulebook_version=_RULEBOOK_VERSION,
        scan_runs=scan_runs,
        deterministic=deterministic,
        games_total=len(frozen_groups),
        completed_games=first.completed_games,
        fail_closed_games=first.fail_closed_games,
        transitions=first.transitions,
        field_audits=first.field_audits,
        lifecycle_counts=first.lifecycle_counts,
        quality_flag_counts=first.quality_flag_counts,
        canonical_state_sha256=first.canonical_state_sha256,
        reducer_latency=_latency_report(latency_samples),
        failures=first.failures,
    )


def _frozen_manifest_path(store_root: Path) -> Path:
    expected_object, expected_schema = X11_FROZEN_PARTITION_ALLOWLIST[2025]
    directory = (
        store_root
        / "manifests"
        / "source=nflverse"
        / f"dataset={_DATASET_ID}"
        / f"version={_SOURCE_VERSION}"
        / "partition=season-2025"
    )
    matches: list[Path] = []
    observed = tuple(sorted(directory.glob("*.manifest.json")))
    for path in observed:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise NFLSeasonCensusError(
                f"cannot classify frozen NFL manifest: {path}"
            ) from error
        if type(document) is not dict:
            raise NFLSeasonCensusError(
                f"frozen NFL manifest must be an object: {path}"
            )
        if (
            document.get("dataset_id") == _DATASET_ID
            and document.get("license_ref") == X11_LICENSE_REF
            and document.get("license_status") == X11_LICENSE_STATUS
            and document.get("upstream_partition") == "season-2025"
            and document.get("object_kind") == "byte_exact_original"
            and document.get("source_cursor")
            == expected_nflverse_source_cursor(2025)
            and document.get("object_sha256") == expected_object
            and document.get("schema_fingerprint") == expected_schema
        ):
            matches.append(path)
    if len(matches) != 1:
        raise NFLSeasonCensusError(
            "frozen 2025 census requires exactly one governed "
            f"{X11_LICENSE_REF}/{X11_LICENSE_STATUS} manifest; "
            f"found {len(matches)} among {len(observed)}"
        )
    return matches[0]


def run_frozen_nflverse_2025_census(
    *,
    program_root: str | Path,
) -> NFLSeasonCensusReport:
    """Verify the immutable 2025 object and audit all native games twice."""

    root = Path(program_root).resolve()
    store_root = root / "var" / "raw"
    manifest_path = _frozen_manifest_path(store_root)
    try:
        verified = read_verified_static_object(
            manifest_path,
            store_root=store_root,
            program_root=root,
        )
    except StaticStoreError as error:
        raise NFLSeasonCensusError(
            "frozen 2025 nflverse object failed manifest verification"
        ) from error
    record = verified.record
    expected_object, _ = X11_FROZEN_PARTITION_ALLOWLIST[2025]
    recomputed_object = (
        f"{_HASH_PREFIX}{hashlib.sha256(verified.object_bytes).hexdigest()}"
    )
    if (
        record.source != "nflverse"
        or record.dataset != _DATASET_ID
        or record.version != _SOURCE_VERSION
        or record.partition != "season-2025"
        or record.manifest.object_sha256 != expected_object
        or recomputed_object != expected_object
    ):
        raise NFLSeasonCensusError(
            "frozen 2025 manifest/raw SHA or source identity is not canonical"
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(
            pa.BufferReader(verified.object_bytes),
            columns=list(_CENSUS_COLUMNS),
        )
    except (pa.ArrowException, OSError, ValueError) as error:
        raise NFLSeasonCensusError(
            "verified 2025 nflverse object is not the frozen Parquet schema"
        ) from error
    rows = table.to_pylist()
    return census_loaded_nflverse_season(
        rows=rows,
        raw_object_sha256=recomputed_object,
        source_version=_SOURCE_VERSION,
        scan_runs=2,
    )


__all__ = [
    "NFLFieldAudit",
    "NFLSeasonCensusError",
    "NFLSeasonCensusFailure",
    "NFLSeasonCensusReport",
    "census_loaded_nflverse_season",
    "run_frozen_nflverse_2025_census",
]
