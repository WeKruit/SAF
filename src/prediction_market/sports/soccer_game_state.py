"""Immutable StatsBomb event-level soccer game state primitives."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from prediction_market.contracts import EventEnvelopeV0
from prediction_market.sports.event_envelopes import (
    validate_static_sport_observation_bundle,
)


_TIMESTAMP_RE = re.compile(
    r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):"
    r"(?P<second>[0-9]{2})\.(?P<millisecond>[0-9]{3})\Z"
)
_MAX_TOP_LEVEL_FIELDS = 64
_MAX_TEXT_LENGTH = 256
_MAX_SOURCE_COORDINATE_MILLI = 1_000_000
_SOURCE_COORDINATE_OUT_OF_BOUNDS_FLAG = "source_coordinate_out_of_bounds"
_CARD_NAMES = frozenset({"Yellow Card", "Second Yellow", "Red Card"})
_BALL_ACTIONS = frozenset(
    {
        "Ball Receipt*",
        "Carry",
        "Clearance",
        "Dribble",
        "Duel",
        "Goal Keeper",
        "Interception",
        "Miscontrol",
        "Pass",
        "Pressure",
        "Shot",
    }
)
_STOPPAGE_ACTIONS = frozenset(
    {
        "Bad Behaviour",
        "Foul Committed",
        "Half End",
        "Half Start",
        "Match End",
        "Own Goal Against",
        "Own Goal For",
        "Period End",
        "Period Start",
        "Starting XI",
        "Substitution",
    }
)


class SoccerGameStateError(ValueError):
    """A soccer event or state violates the point-in-time state contract."""


def _required_text(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or len(value) > _MAX_TEXT_LENGTH
    ):
        raise SoccerGameStateError(
            f"{field_name} must be a bounded nonempty string"
        )
    return value


def _required_int(
    value: object,
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        upper = "" if maximum is None else f" and <= {maximum}"
        raise SoccerGameStateError(
            f"{field_name} must be an integer >= {minimum}{upper}"
        )
    return value


def _required_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if type(value) is not dict or len(value) > _MAX_TOP_LEVEL_FIELDS:
        raise SoccerGameStateError(f"{field_name} must be a bounded object")
    return value


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return [_canonical_value(child) for child in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    raise SoccerGameStateError(
        f"canonical material contains unsupported {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class SoccerTeamPlayers:
    """The active players for one team, in stable lineup order."""

    team_id: int
    player_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _required_int(self.team_id, "active_players.team_id", minimum=1)
        if (
            type(self.player_ids) is not tuple
            or len(self.player_ids) > 11
            or any(
                type(player_id) is not int or player_id <= 0
                for player_id in self.player_ids
            )
            or len(set(self.player_ids)) != len(self.player_ids)
        ):
            raise SoccerGameStateError(
                "active player ids must be a tuple of at most 11 unique positive integers"
            )


@dataclass(frozen=True, slots=True)
class SoccerCard:
    """One card issued at a known event sequence."""

    sequence: int
    team_id: int
    player_id: int
    card: str

    def __post_init__(self) -> None:
        _required_int(self.sequence, "card.sequence", minimum=1)
        _required_int(self.team_id, "card.team_id", minimum=1)
        _required_int(self.player_id, "card.player_id", minimum=1)
        if self.card not in _CARD_NAMES:
            raise SoccerGameStateError("card is not a supported StatsBomb card")


@dataclass(frozen=True, slots=True)
class SoccerSubstitution:
    """One completed player-for-player substitution."""

    sequence: int
    team_id: int
    player_out_id: int
    player_in_id: int

    def __post_init__(self) -> None:
        _required_int(self.sequence, "substitution.sequence", minimum=1)
        _required_int(self.team_id, "substitution.team_id", minimum=1)
        _required_int(
            self.player_out_id, "substitution.player_out_id", minimum=1
        )
        _required_int(
            self.player_in_id, "substitution.player_in_id", minimum=1
        )
        if self.player_out_id == self.player_in_id:
            raise SoccerGameStateError(
                "substitution players must be different"
            )


@dataclass(frozen=True, slots=True)
class SoccerPlayerOff:
    """One observed temporary or permanent player absence."""

    sequence: int
    team_id: int
    player_id: int
    roster_index: int
    permanent: bool

    def __post_init__(self) -> None:
        _required_int(self.sequence, "player_off.sequence", minimum=1)
        _required_int(self.team_id, "player_off.team_id", minimum=1)
        _required_int(self.player_id, "player_off.player_id", minimum=1)
        _required_int(
            self.roster_index,
            "player_off.roster_index",
            maximum=10,
        )
        if type(self.permanent) is not bool:
            raise SoccerGameStateError("player_off.permanent must be boolean")


@dataclass(frozen=True, slots=True)
class SoccerGameEvent:
    """One normalized event containing only information known at that event."""

    game_id: str
    sequence: int
    native_event_id: str
    period: int
    clock_ms: int
    period_clock_ms: int
    action: str
    team_id: int
    event_id: str
    lineup_player_ids: tuple[int, ...] = ()
    player_id: int | None = None
    possession_id: int | None = None
    possession_team_id: int | None = None
    play_pattern: str | None = None
    ball_x_milli: int | None = None
    ball_y_milli: int | None = None
    end_ball_x_milli: int | None = None
    end_ball_y_milli: int | None = None
    pass_outcome: str | None = None
    shot_outcome: str | None = None
    card: str | None = None
    replacement_player_id: int | None = None
    player_off_permanent: bool | None = None
    score_for_team_id: int | None = None
    in_play: bool | None = None
    quality_flags: tuple[str, ...] = ()
    sport: str = field(default="soccer", init=False)

    def __post_init__(self) -> None:
        _required_text(self.game_id, "game_id")
        _required_int(self.sequence, "sequence", minimum=1)
        _required_text(self.native_event_id, "native_event_id")
        _required_int(self.period, "period", minimum=1, maximum=5)
        _required_int(self.clock_ms, "clock_ms")
        _required_int(self.period_clock_ms, "period_clock_ms")
        _required_text(self.action, "action")
        _required_int(self.team_id, "team_id", minimum=1)
        if re.fullmatch(r"evt_[0-9a-f]{64}", self.event_id) is None:
            raise SoccerGameStateError(
                "event_id must be an external EventEnvelope evt_<sha256>"
            )
        if type(self.lineup_player_ids) is not tuple or (
            len(self.lineup_player_ids) > 25
            or any(
                type(player_id) is not int or player_id <= 0
                for player_id in self.lineup_player_ids
            )
            or len(set(self.lineup_player_ids)) != len(self.lineup_player_ids)
        ):
            raise SoccerGameStateError(
                "lineup_player_ids must be a bounded tuple of unique positive integers"
            )
        if self.action == "Starting XI" and len(self.lineup_player_ids) != 11:
            raise SoccerGameStateError(
                "Starting XI must contain exactly 11 player ids"
            )
        if self.action != "Starting XI" and self.lineup_player_ids:
            raise SoccerGameStateError(
                "only Starting XI can contain lineup_player_ids"
            )
        for field_name in (
            "player_id",
            "possession_id",
            "possession_team_id",
            "replacement_player_id",
            "score_for_team_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _required_int(value, field_name, minimum=1)
        for field_name in (
            "play_pattern",
            "pass_outcome",
            "shot_outcome",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _required_text(value, field_name)
        if (
            type(self.quality_flags) is not tuple
            or len(self.quality_flags) != len(set(self.quality_flags))
            or tuple(sorted(self.quality_flags)) != self.quality_flags
        ):
            raise SoccerGameStateError(
                "quality_flags must be a sorted unique tuple"
            )
        for quality_flag in self.quality_flags:
            _required_text(quality_flag, "quality_flags[]")
        allow_source_out_of_bounds = (
            _SOURCE_COORDINATE_OUT_OF_BOUNDS_FLAG in self.quality_flags
        )
        has_source_out_of_bounds = False
        for x_name, y_name, x_maximum, y_maximum in (
            ("ball_x_milli", "ball_y_milli", 120_000, 80_000),
            (
                "end_ball_x_milli",
                "end_ball_y_milli",
                120_000,
                80_000,
            ),
        ):
            x_value = getattr(self, x_name)
            y_value = getattr(self, y_name)
            if (x_value is None) != (y_value is None):
                raise SoccerGameStateError(
                    f"{x_name} and {y_name} must be supplied together"
                )
            if x_value is not None and y_value is not None:
                if allow_source_out_of_bounds:
                    _required_int(
                        x_value,
                        x_name,
                        minimum=-_MAX_SOURCE_COORDINATE_MILLI,
                        maximum=_MAX_SOURCE_COORDINATE_MILLI,
                    )
                    _required_int(
                        y_value,
                        y_name,
                        minimum=-_MAX_SOURCE_COORDINATE_MILLI,
                        maximum=_MAX_SOURCE_COORDINATE_MILLI,
                    )
                else:
                    _required_int(x_value, x_name, maximum=x_maximum)
                    _required_int(y_value, y_name, maximum=y_maximum)
                has_source_out_of_bounds = has_source_out_of_bounds or (
                    x_value < 0
                    or x_value > x_maximum
                    or y_value < 0
                    or y_value > y_maximum
                )
        if allow_source_out_of_bounds and not has_source_out_of_bounds:
            raise SoccerGameStateError(
                "source_coordinate_out_of_bounds requires an out-of-bounds "
                "event coordinate"
            )
        if self.card is not None and self.card not in _CARD_NAMES:
            raise SoccerGameStateError("card is not a supported StatsBomb card")
        if self.card is not None and self.player_id is None:
            raise SoccerGameStateError("card events require player_id")
        if self.replacement_player_id is not None:
            if self.action != "Substitution" or self.player_id is None:
                raise SoccerGameStateError(
                    "substitution replacement requires an outgoing player"
                )
            if self.replacement_player_id == self.player_id:
                raise SoccerGameStateError(
                    "substitution replacement must differ from outgoing player"
                )
        if self.action == "Player Off":
            if (
                self.player_id is None
                or type(self.player_off_permanent) is not bool
            ):
                raise SoccerGameStateError(
                    "Player Off requires player_id and player_off_permanent"
                )
        elif self.player_off_permanent is not None:
            raise SoccerGameStateError(
                "only Player Off can contain player_off_permanent"
            )
        if self.action == "Player On" and self.player_id is None:
            raise SoccerGameStateError("Player On requires player_id")
        if self.score_for_team_id is not None and not (
            (self.action == "Shot" and self.shot_outcome == "Goal")
            or self.action == "Own Goal For"
        ):
            raise SoccerGameStateError(
                "score_for_team_id requires a goal shot or Own Goal For"
            )
        if self.in_play is not None and type(self.in_play) is not bool:
            raise SoccerGameStateError("in_play must be a boolean or null")


@dataclass(frozen=True, slots=True)
class SoccerGameState:
    """A complete point-in-time soccer state produced by event reduction."""

    game_id: str
    home_team_id: int
    away_team_id: int
    sequence: int = 0
    period: int = 0
    clock_ms: int = 0
    period_clock_ms: int = 0
    home_score: int = 0
    away_score: int = 0
    possession_id: int | None = None
    possession_team_id: int | None = None
    play_pattern: str | None = None
    ball_x_milli: int | None = None
    ball_y_milli: int | None = None
    in_play: bool = False
    last_action: str | None = None
    last_event_id: str | None = None
    cards: tuple[SoccerCard, ...] = ()
    substitutions: tuple[SoccerSubstitution, ...] = ()
    players_off: tuple[SoccerPlayerOff, ...] = ()
    active_players: tuple[SoccerTeamPlayers, ...] = ()
    quality_flags: tuple[str, ...] = ()
    terminal: bool = False
    terminal_reason: str | None = None
    sport: str = field(default="soccer", init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _required_text(self.game_id, "game_id")
        _required_int(self.home_team_id, "home_team_id", minimum=1)
        _required_int(self.away_team_id, "away_team_id", minimum=1)
        if self.home_team_id == self.away_team_id:
            raise SoccerGameStateError(
                "home_team_id and away_team_id must differ"
            )
        _required_int(self.sequence, "sequence")
        _required_int(self.period, "period", maximum=5)
        _required_int(self.clock_ms, "clock_ms")
        _required_int(self.period_clock_ms, "period_clock_ms")
        _required_int(self.home_score, "home_score")
        _required_int(self.away_score, "away_score")
        if self.possession_id is not None:
            _required_int(self.possession_id, "possession_id", minimum=1)
        if self.possession_team_id is not None:
            _required_int(
                self.possession_team_id,
                "possession_team_id",
                minimum=1,
            )
            if self.possession_team_id not in self.team_ids:
                raise SoccerGameStateError(
                    "possession_team_id is not a game team"
                )
        if self.play_pattern is not None:
            _required_text(self.play_pattern, "play_pattern")
        allow_source_out_of_bounds = (
            type(self.quality_flags) is tuple
            and _SOURCE_COORDINATE_OUT_OF_BOUNDS_FLAG in self.quality_flags
        )
        if (self.ball_x_milli is None) != (self.ball_y_milli is None):
            raise SoccerGameStateError(
                "ball_x_milli and ball_y_milli must be supplied together"
            )
        if self.ball_x_milli is not None and self.ball_y_milli is not None:
            if allow_source_out_of_bounds:
                _required_int(
                    self.ball_x_milli,
                    "ball_x_milli",
                    minimum=-_MAX_SOURCE_COORDINATE_MILLI,
                    maximum=_MAX_SOURCE_COORDINATE_MILLI,
                )
                _required_int(
                    self.ball_y_milli,
                    "ball_y_milli",
                    minimum=-_MAX_SOURCE_COORDINATE_MILLI,
                    maximum=_MAX_SOURCE_COORDINATE_MILLI,
                )
            else:
                _required_int(
                    self.ball_x_milli, "ball_x_milli", maximum=120_000
                )
                _required_int(
                    self.ball_y_milli, "ball_y_milli", maximum=80_000
                )
        if type(self.in_play) is not bool:
            raise SoccerGameStateError("in_play must be a boolean")
        if self.last_action is not None:
            _required_text(self.last_action, "last_action")
        if self.last_event_id is not None and re.fullmatch(
            r"evt_[0-9a-f]{64}", self.last_event_id
        ) is None:
            raise SoccerGameStateError(
                "last_event_id must be a canonical event id"
            )
        if type(self.cards) is not tuple or any(
            not isinstance(card, SoccerCard) for card in self.cards
        ):
            raise SoccerGameStateError("cards must be a tuple of SoccerCard")
        if type(self.substitutions) is not tuple or any(
            not isinstance(substitution, SoccerSubstitution)
            for substitution in self.substitutions
        ):
            raise SoccerGameStateError(
                "substitutions must be a tuple of SoccerSubstitution"
            )
        if type(self.players_off) is not tuple or any(
            not isinstance(player_off, SoccerPlayerOff)
            for player_off in self.players_off
        ):
            raise SoccerGameStateError(
                "players_off must be a tuple of SoccerPlayerOff"
            )
        if (
            type(self.quality_flags) is not tuple
            or len(self.quality_flags) != len(set(self.quality_flags))
            or tuple(sorted(self.quality_flags)) != self.quality_flags
        ):
            raise SoccerGameStateError(
                "quality_flags must be a sorted unique tuple"
            )
        for quality_flag in self.quality_flags:
            _required_text(quality_flag, "quality_flags[]")
        active_players = self.active_players
        if active_players == ():
            active_players = (
                SoccerTeamPlayers(self.home_team_id, ()),
                SoccerTeamPlayers(self.away_team_id, ()),
            )
            object.__setattr__(self, "active_players", active_players)
        expected_team_ids = (self.home_team_id, self.away_team_id)
        if (
            type(active_players) is not tuple
            or tuple(team.team_id for team in active_players)
            != expected_team_ids
        ):
            raise SoccerGameStateError(
                "active_players must contain home then away team"
            )
        all_active_ids = tuple(
            player_id
            for team in active_players
            for player_id in team.player_ids
        )
        if len(all_active_ids) != len(set(all_active_ids)):
            raise SoccerGameStateError(
                "a player cannot be active for both teams"
            )
        off_keys = tuple(
            (player_off.team_id, player_off.player_id)
            for player_off in self.players_off
        )
        if len(off_keys) != len(set(off_keys)):
            raise SoccerGameStateError(
                "a player can have at most one active Player Off record"
            )
        for player_off in self.players_off:
            if player_off.team_id not in self.team_ids:
                raise SoccerGameStateError(
                    "player_off team is not a game team"
                )
            if player_off.player_id in all_active_ids:
                raise SoccerGameStateError(
                    "a Player Off player cannot remain active"
                )
        if self.terminal != (self.terminal_reason is not None):
            raise SoccerGameStateError(
                "terminal and terminal_reason must agree"
            )
        material = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "state_sha256"
        }
        digest = hashlib.sha256(_canonical_bytes(material)).hexdigest()
        object.__setattr__(self, "state_sha256", f"sha256:{digest}")

    @property
    def team_ids(self) -> tuple[int, int]:
        return self.home_team_id, self.away_team_id


def _event_clocks(raw_event: Mapping[str, object]) -> tuple[int, int]:
    minute = _required_int(
        raw_event.get("minute"), "minute", minimum=0, maximum=300
    )
    second = _required_int(
        raw_event.get("second"), "second", minimum=0, maximum=59
    )
    timestamp = _required_text(raw_event.get("timestamp"), "timestamp")
    match = _TIMESTAMP_RE.fullmatch(timestamp)
    if match is None:
        raise SoccerGameStateError(
            "timestamp must use StatsBomb HH:MM:SS.mmm format"
        )
    timestamp_hour = int(match.group("hour"))
    timestamp_minute = int(match.group("minute"))
    timestamp_second = int(match.group("second"))
    millisecond = int(match.group("millisecond"))
    if timestamp_minute > 59 or timestamp_second > 59:
        raise SoccerGameStateError("timestamp components are out of range")
    if timestamp_second != second:
        raise SoccerGameStateError("timestamp second must match event second")
    period_clock_ms = (
        ((timestamp_hour * 60 + timestamp_minute) * 60 + timestamp_second)
        * 1_000
        + millisecond
    )
    clock_ms = (minute * 60 + second) * 1_000 + millisecond
    return clock_ms, period_clock_ms


def _starting_lineup(
    raw_event: Mapping[str, object],
    *,
    action: str,
) -> tuple[int, ...]:
    if action != "Starting XI":
        return ()
    tactics = _required_mapping(raw_event.get("tactics"), "tactics")
    lineup = tactics.get("lineup")
    if type(lineup) is not list or len(lineup) != 11:
        raise SoccerGameStateError("Starting XI must contain exactly 11 players")
    player_ids: list[int] = []
    for position, item in enumerate(lineup):
        lineup_item = _required_mapping(item, f"lineup[{position}]")
        player = _required_mapping(
            lineup_item.get("player"), f"lineup[{position}].player"
        )
        player_ids.append(
            _required_int(
                player.get("id"),
                f"lineup[{position}].player.id",
                minimum=1,
            )
        )
    if len(set(player_ids)) != len(player_ids):
        raise SoccerGameStateError("Starting XI player ids must be unique")
    return tuple(player_ids)


def _optional_entity_id(
    raw_event: Mapping[str, object],
    field_name: str,
) -> int | None:
    value = raw_event.get(field_name)
    if value is None:
        return None
    entity = _required_mapping(value, field_name)
    return _required_int(entity.get("id"), f"{field_name}.id", minimum=1)


def _optional_entity_name(
    parent: Mapping[str, object],
    field_name: str,
) -> str | None:
    value = parent.get(field_name)
    if value is None:
        return None
    entity = _required_mapping(value, field_name)
    return _required_text(entity.get("name"), f"{field_name}.name")


def _coordinate_milli(
    value: object,
    field_name: str,
    *,
    maximum: int,
    allow_source_out_of_bounds: bool,
) -> int:
    if type(value) not in {int, float}:
        raise SoccerGameStateError(f"{field_name} must be a JSON number")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as exc:
        raise SoccerGameStateError(f"{field_name} must be finite") from exc
    if not decimal.is_finite():
        raise SoccerGameStateError(f"{field_name} must be finite")
    scaled = decimal * 1_000
    if scaled != scaled.to_integral_value():
        raise SoccerGameStateError(
            f"{field_name} must have at most three decimal places"
        )
    result = int(scaled)
    if allow_source_out_of_bounds:
        return _required_int(
            result,
            field_name,
            minimum=-_MAX_SOURCE_COORDINATE_MILLI,
            maximum=_MAX_SOURCE_COORDINATE_MILLI,
        )
    return _required_int(result, field_name, maximum=maximum)


def _optional_location(
    value: object,
    field_name: str,
    *,
    allow_source_out_of_bounds: bool,
) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    if type(value) is not list or len(value) not in {2, 3}:
        raise SoccerGameStateError(
            f"{field_name} must contain two or three coordinates"
        )
    return (
        _coordinate_milli(
            value[0],
            f"{field_name}[0]",
            maximum=120_000,
            allow_source_out_of_bounds=allow_source_out_of_bounds,
        ),
        _coordinate_milli(
            value[1],
            f"{field_name}[1]",
            maximum=80_000,
            allow_source_out_of_bounds=allow_source_out_of_bounds,
        ),
    )


def _action_details(
    raw_event: Mapping[str, object],
    *,
    action: str,
    team_id: int,
    allow_source_out_of_bounds: bool,
) -> tuple[
    int | None,
    int | None,
    str | None,
    str | None,
    str | None,
    int | None,
    int | None,
    bool | None,
]:
    end_x: int | None = None
    end_y: int | None = None
    pass_outcome: str | None = None
    shot_outcome: str | None = None
    card: str | None = None
    replacement_player_id: int | None = None
    score_for_team_id: int | None = None

    if action in {"Carry", "Goal Keeper", "Pass"}:
        detail_name = {
            "Carry": "carry",
            "Goal Keeper": "goalkeeper",
            "Pass": "pass",
        }[action]
        action_detail = _required_mapping(
            raw_event.get(detail_name),
            detail_name,
        )
        end_x, end_y = _optional_location(
            action_detail.get("end_location"),
            f"{detail_name}.end_location",
            allow_source_out_of_bounds=allow_source_out_of_bounds,
        )
        if action == "Pass":
            pass_outcome = _optional_entity_name(action_detail, "outcome")
    elif action == "Shot":
        shot = _required_mapping(raw_event.get("shot"), "shot")
        end_x, end_y = _optional_location(
            shot.get("end_location"),
            "shot.end_location",
            allow_source_out_of_bounds=allow_source_out_of_bounds,
        )
        shot_outcome = _optional_entity_name(shot, "outcome")
        if shot_outcome is None:
            raise SoccerGameStateError("Shot requires shot.outcome")
        if shot_outcome == "Goal":
            score_for_team_id = team_id
    elif action == "Substitution":
        substitution = _required_mapping(
            raw_event.get("substitution"), "substitution"
        )
        replacement = _required_mapping(
            substitution.get("replacement"), "substitution.replacement"
        )
        replacement_player_id = _required_int(
            replacement.get("id"),
            "substitution.replacement.id",
            minimum=1,
        )
    elif action == "Own Goal For":
        score_for_team_id = team_id

    card_object_name = {
        "Bad Behaviour": "bad_behaviour",
        "Foul Committed": "foul_committed",
    }.get(action)
    if card_object_name is not None:
        detail_value = raw_event.get(card_object_name)
        detail = (
            None
            if detail_value is None
            else _required_mapping(detail_value, card_object_name)
        )
        card = (
            None
            if detail is None
            else _optional_entity_name(detail, "card")
        )
    if card is not None:
        if card not in _CARD_NAMES:
            raise SoccerGameStateError(
                f"unsupported StatsBomb card: {card}"
            )

    if action == "Pass":
        in_play = pass_outcome not in {"Out", "Pass Offside"}
    elif action == "Shot":
        in_play = shot_outcome in {"Blocked", "Saved", "Saved to Post"}
    elif action in _STOPPAGE_ACTIONS:
        in_play = False
    elif action in _BALL_ACTIONS:
        in_play = True
    else:
        in_play = None

    return (
        end_x,
        end_y,
        pass_outcome,
        shot_outcome,
        card,
        replacement_player_id,
        score_for_team_id,
        in_play,
    )


def statsbomb_event_payload(
    raw_event: Mapping[str, object],
    *,
    game_id: str,
    quality_flags: tuple[str, ...] = (),
) -> dict[str, object]:
    """Normalize one StatsBomb event into a complete envelope payload."""

    event = _required_mapping(raw_event, "StatsBomb event")
    if (
        type(quality_flags) is not tuple
        or len(quality_flags) != len(set(quality_flags))
    ):
        raise SoccerGameStateError("quality_flags must be a unique tuple")
    allow_source_out_of_bounds = (
        _SOURCE_COORDINATE_OUT_OF_BOUNDS_FLAG in quality_flags
    )
    event_type = _required_mapping(event.get("type"), "type")
    team = _required_mapping(event.get("team"), "team")
    action = _required_text(event_type.get("name"), "type.name")
    clock_ms, period_clock_ms = _event_clocks(event)
    team_id = _required_int(team.get("id"), "team.id", minimum=1)
    possession = event.get("possession")
    possession_id = (
        None
        if possession is None
        else _required_int(possession, "possession", minimum=1)
    )
    possession_team_id = _optional_entity_id(event, "possession_team")
    if possession_id is None and possession_team_id is not None:
        raise SoccerGameStateError(
            "possession_team requires a possession identifier"
        )
    play_pattern = _optional_entity_name(event, "play_pattern")
    player_off_permanent: bool | None = None
    if action == "Player Off":
        player_off = event.get("player_off")
        if player_off is None:
            player_off_permanent = False
        else:
            player_off_detail = _required_mapping(
                player_off,
                "player_off",
            )
            permanent = player_off_detail.get("permanent", False)
            if type(permanent) is not bool:
                raise SoccerGameStateError(
                    "player_off.permanent must be boolean"
                )
            player_off_permanent = permanent
    ball_x, ball_y = _optional_location(
        event.get("location"),
        "location",
        allow_source_out_of_bounds=allow_source_out_of_bounds,
    )
    (
        end_x,
        end_y,
        pass_outcome,
        shot_outcome,
        card,
        replacement_player_id,
        score_for_team_id,
        in_play,
    ) = _action_details(
        event,
        action=action,
        team_id=team_id,
        allow_source_out_of_bounds=allow_source_out_of_bounds,
    )
    has_source_out_of_bounds = any(
        coordinate is not None
        and (
            coordinate < 0
            or coordinate > maximum
        )
        for coordinate, maximum in (
            (ball_x, 120_000),
            (ball_y, 80_000),
            (end_x, 120_000),
            (end_y, 80_000),
        )
    )
    if allow_source_out_of_bounds and not has_source_out_of_bounds:
        raise SoccerGameStateError(
            "source_coordinate_out_of_bounds requires an out-of-bounds "
            "event coordinate"
        )
    return {
        "sport": "soccer",
        "game_id": _required_text(game_id, "game_id"),
        "sequence": _required_int(event.get("index"), "index", minimum=1),
        "native_event_id": _required_text(event.get("id"), "id"),
        "period": _required_int(
            event.get("period"),
            "period",
            minimum=1,
            maximum=5,
        ),
        "clock_ms": clock_ms,
        "period_clock_ms": period_clock_ms,
        "action": action,
        "team_id": team_id,
        "lineup_player_ids": list(_starting_lineup(event, action=action)),
        "player_id": _optional_entity_id(event, "player"),
        "possession_id": possession_id,
        "possession_team_id": possession_team_id,
        "play_pattern": play_pattern,
        "ball_x_milli": ball_x,
        "ball_y_milli": ball_y,
        "end_ball_x_milli": end_x,
        "end_ball_y_milli": end_y,
        "pass_outcome": pass_outcome,
        "shot_outcome": shot_outcome,
        "card": card,
        "replacement_player_id": replacement_player_id,
        "player_off_permanent": player_off_permanent,
        "score_for_team_id": score_for_team_id,
        "in_play": in_play,
        "quality_flags": list(sorted(quality_flags)),
    }


def adapt_statsbomb_event(
    envelope: EventEnvelopeV0,
    *,
    program_root: str | Path,
    raw_parents: tuple[EventEnvelopeV0, ...],
) -> SoccerGameEvent:
    """Construct a soccer event only from a fully bound normalized envelope."""

    validated = validate_static_sport_observation_bundle(
        program_root,
        envelope,
        raw_parents=raw_parents,
        expected_experiment_id="X-12",
        expected_dataset_id="DS-STATSBOMB-OPEN",
        expected_source_system="statsbomb",
        expected_source_stream="events",
        expected_native_namespace="statsbomb.event",
    )
    if len(raw_parents) != 1:
        raise SoccerGameStateError(
            "a StatsBomb event requires exactly one raw row parent"
        )
    payload = dict(validated.payload)
    expected_payload_fields = {
        item.name for item in fields(SoccerGameEvent)
    } - {"event_id"}
    if set(payload) != expected_payload_fields:
        raise SoccerGameStateError(
            "normalized StatsBomb payload fields are incomplete or unexpected"
        )
    if payload.pop("sport") != "soccer":
        raise SoccerGameStateError(
            "normalized StatsBomb payload sport must be soccer"
        )
    canonical_game_id = validated.canonical_refs.game_id
    if (
        canonical_game_id is None
        or not canonical_game_id.startswith("game_statsbomb_")
        or payload["game_id"] != canonical_game_id
    ):
        raise SoccerGameStateError(
            "normalized StatsBomb envelope game identity is invalid"
        )
    raw_parent = raw_parents[0]
    if raw_parent.lineage.raw_record_ordinal != payload["sequence"] - 1:
        raise SoccerGameStateError(
            "normalized StatsBomb sequence does not match raw row ordinal"
        )
    if raw_parent.native_refs[0].native_id != payload["native_event_id"]:
        raise SoccerGameStateError(
            "normalized StatsBomb event does not match raw native identity"
        )
    payload["lineup_player_ids"] = tuple(payload["lineup_player_ids"])
    payload["quality_flags"] = tuple(payload["quality_flags"])
    return SoccerGameEvent(
        event_id=validated.event_id,
        **payload,
    )


def initial_soccer_game_state(
    game_id: str,
    *,
    home_team_id: int,
    away_team_id: int,
) -> SoccerGameState:
    """Create the immutable pre-event state for one known fixture."""

    return SoccerGameState(
        game_id=game_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )


def _replace_team_players(
    active_players: tuple[SoccerTeamPlayers, ...],
    *,
    team_id: int,
    player_ids: tuple[int, ...],
) -> tuple[SoccerTeamPlayers, ...]:
    return tuple(
        SoccerTeamPlayers(team.team_id, player_ids)
        if team.team_id == team_id
        else team
        for team in active_players
    )


def _active_ids_for(
    state: SoccerGameState,
    team_id: int,
) -> tuple[int, ...]:
    for team in state.active_players:
        if team.team_id == team_id:
            return team.player_ids
    raise SoccerGameStateError("event team is not present in active_players")


def reduce_soccer_game_state(
    state: SoccerGameState,
    event: SoccerGameEvent,
) -> SoccerGameState:
    """Apply one event to one state, returning a new immutable state."""

    if not isinstance(state, SoccerGameState):
        raise SoccerGameStateError("state must be SoccerGameState")
    if not isinstance(event, SoccerGameEvent):
        raise SoccerGameStateError("event must be SoccerGameEvent")
    if state.terminal:
        raise SoccerGameStateError("terminal state cannot accept more events")
    if event.game_id != state.game_id:
        raise SoccerGameStateError("event game_id does not match state game_id")
    if event.sequence != state.sequence + 1:
        raise SoccerGameStateError(
            "event sequence must be contiguous and increasing"
        )
    if event.team_id not in state.team_ids:
        raise SoccerGameStateError("event team_id is not a game team")
    if (
        event.possession_team_id is not None
        and event.possession_team_id not in state.team_ids
    ):
        raise SoccerGameStateError(
            "event possession_team_id is not a game team"
        )
    if event.period < state.period:
        raise SoccerGameStateError("event period cannot regress")
    if event.period > state.period + 1:
        raise SoccerGameStateError("event period cannot skip a period")
    same_period_clock_regression = (
        event.period == state.period
        and (
            event.clock_ms < state.clock_ms
            or event.period_clock_ms < state.period_clock_ms
        )
    )
    if same_period_clock_regression and not {
        "clock_jump",
        "out_of_order",
    }.issubset(event.quality_flags):
        raise SoccerGameStateError(
            "same-period clock regression requires clock_jump and "
            "out_of_order quality flags"
        )
    if event.period > state.period and state.period > 0 and event.action not in {
        "Half Start",
        "Period Start",
    }:
        raise SoccerGameStateError(
            "a new period requires an explicit period start event"
        )
    if (
        event.action in {"Half Start", "Period Start"}
        and event.period_clock_ms != 0
    ):
        raise SoccerGameStateError(
            "period start event must have a zero period clock"
        )
    if event.action == "Match End" and event.period < 2:
        raise SoccerGameStateError(
            "Match End cannot occur before the second period"
        )

    active_players = state.active_players
    if event.action == "Starting XI":
        current_ids = _active_ids_for(state, event.team_id)
        if current_ids:
            raise SoccerGameStateError(
                "Starting XI cannot replace an initialized lineup"
            )
        active_players = _replace_team_players(
            active_players,
            team_id=event.team_id,
            player_ids=event.lineup_player_ids,
        )

    players_off = state.players_off
    if event.action == "Player Off":
        assert event.player_id is not None
        assert event.player_off_permanent is not None
        current_ids = next(
            team.player_ids
            for team in active_players
            if team.team_id == event.team_id
        )
        if event.player_id not in current_ids:
            raise SoccerGameStateError("Player Off player is not active")
        current_index = current_ids.index(event.player_id)
        roster_index = current_index
        for existing_index in sorted(
            player_off.roster_index
            for player_off in players_off
            if player_off.team_id == event.team_id
        ):
            if existing_index <= roster_index:
                roster_index += 1
        players_off = players_off + (
            SoccerPlayerOff(
                sequence=event.sequence,
                team_id=event.team_id,
                player_id=event.player_id,
                roster_index=roster_index,
                permanent=event.player_off_permanent,
            ),
        )
        active_players = _replace_team_players(
            active_players,
            team_id=event.team_id,
            player_ids=tuple(
                player_id
                for player_id in current_ids
                if player_id != event.player_id
            ),
        )
    elif event.action == "Player On":
        assert event.player_id is not None
        matches = tuple(
            player_off
            for player_off in players_off
            if player_off.team_id == event.team_id
            and player_off.player_id == event.player_id
        )
        if len(matches) != 1:
            raise SoccerGameStateError(
                "Player On requires one matching Player Off record"
            )
        absence = matches[0]
        if absence.permanent:
            raise SoccerGameStateError(
                "a permanent Player Off cannot return with Player On"
            )
        current_ids = next(
            team.player_ids
            for team in active_players
            if team.team_id == event.team_id
        )
        insertion_index = absence.roster_index - sum(
            player_off.team_id == event.team_id
            and player_off.roster_index < absence.roster_index
            for player_off in players_off
            if player_off != absence
        )
        if insertion_index < 0 or insertion_index > len(current_ids):
            raise SoccerGameStateError(
                "Player On roster position is inconsistent"
            )
        restored_ids = (
            current_ids[:insertion_index]
            + (event.player_id,)
            + current_ids[insertion_index:]
        )
        active_players = _replace_team_players(
            active_players,
            team_id=event.team_id,
            player_ids=restored_ids,
        )
        players_off = tuple(
            player_off
            for player_off in players_off
            if player_off != absence
        )

    cards = state.cards
    if event.card is not None:
        assert event.player_id is not None
        cards = cards + (
            SoccerCard(
                sequence=event.sequence,
                team_id=event.team_id,
                player_id=event.player_id,
                card=event.card,
            ),
        )
        if event.card in {"Second Yellow", "Red Card"}:
            current_ids = _active_ids_for(state, event.team_id)
            if event.player_id in current_ids:
                active_players = _replace_team_players(
                    active_players,
                    team_id=event.team_id,
                    player_ids=tuple(
                        player_id
                        for player_id in current_ids
                        if player_id != event.player_id
                    ),
                )

    substitutions = state.substitutions
    if event.replacement_player_id is not None:
        assert event.player_id is not None
        current_ids = next(
            team.player_ids
            for team in active_players
            if team.team_id == event.team_id
        )
        if event.player_id not in current_ids:
            raise SoccerGameStateError(
                "substitution outgoing player is not active"
            )
        if any(
            event.replacement_player_id in team.player_ids
            for team in active_players
        ):
            raise SoccerGameStateError(
                "substitution replacement player is already active"
            )
        active_players = _replace_team_players(
            active_players,
            team_id=event.team_id,
            player_ids=tuple(
                event.replacement_player_id
                if player_id == event.player_id
                else player_id
                for player_id in current_ids
            ),
        )
        substitutions = substitutions + (
            SoccerSubstitution(
                sequence=event.sequence,
                team_id=event.team_id,
                player_out_id=event.player_id,
                player_in_id=event.replacement_player_id,
            ),
        )

    home_score = state.home_score
    away_score = state.away_score
    if event.score_for_team_id is not None:
        if event.score_for_team_id not in state.team_ids:
            raise SoccerGameStateError(
                "score_for_team_id is not a game team"
            )
        if event.score_for_team_id == state.home_team_id:
            home_score += 1
        else:
            away_score += 1

    ball_x_milli = state.ball_x_milli
    ball_y_milli = state.ball_y_milli
    if event.end_ball_x_milli is not None:
        ball_x_milli = event.end_ball_x_milli
        ball_y_milli = event.end_ball_y_milli
    elif event.ball_x_milli is not None:
        ball_x_milli = event.ball_x_milli
        ball_y_milli = event.ball_y_milli

    terminal = event.action == "Match End"
    return replace(
        state,
        sequence=event.sequence,
        period=event.period,
        clock_ms=event.clock_ms,
        period_clock_ms=event.period_clock_ms,
        home_score=home_score,
        away_score=away_score,
        possession_id=(
            event.possession_id
            if event.possession_id is not None
            else state.possession_id
        ),
        possession_team_id=(
            event.possession_team_id
            if event.possession_id is not None
            else state.possession_team_id
        ),
        play_pattern=(
            event.play_pattern
            if event.play_pattern is not None
            else state.play_pattern
        ),
        ball_x_milli=ball_x_milli,
        ball_y_milli=ball_y_milli,
        in_play=(
            event.in_play if event.in_play is not None else state.in_play
        ),
        last_action=event.action,
        last_event_id=event.event_id,
        cards=cards,
        substitutions=substitutions,
        players_off=players_off,
        active_players=active_players,
        quality_flags=tuple(
            sorted(set(state.quality_flags).union(event.quality_flags))
        ),
        terminal=terminal,
        terminal_reason="match_end" if terminal else None,
    )


# Short aliases for protocol consumers that dispatch reducers structurally.
reduce = reduce_soccer_game_state


@dataclass(frozen=True, slots=True)
class SoccerGameStateReducer:
    """Common-protocol adapter for the pure soccer reducer."""

    sport: str = field(default="soccer", init=False)
    reducer_id: str = field(
        default="soccer.statsbomb.event-reducer",
        init=False,
    )
    reducer_version: str = field(default="v2", init=False)

    def reduce(
        self,
        state: SoccerGameState,
        event: SoccerGameEvent,
    ) -> SoccerGameState:
        return reduce_soccer_game_state(state, event)


SOCCER_GAME_STATE_REDUCER = SoccerGameStateReducer()


__all__ = [
    "SOCCER_GAME_STATE_REDUCER",
    "SoccerCard",
    "SoccerGameEvent",
    "SoccerGameState",
    "SoccerGameStateError",
    "SoccerGameStateReducer",
    "SoccerPlayerOff",
    "SoccerSubstitution",
    "SoccerTeamPlayers",
    "adapt_statsbomb_event",
    "initial_soccer_game_state",
    "reduce",
    "reduce_soccer_game_state",
    "statsbomb_event_payload",
]
