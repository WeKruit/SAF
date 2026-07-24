"""Full-season, same-source MLB reducer census over frozen Retrosheet bytes."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from prediction_market.contracts import canonical_json, canonical_sha256
from prediction_market.static_store import read_verified_static_object
from prediction_market.sports import mlb_game_state as mlb


EXPECTED_EVENT_FILES = 30
EXPECTED_GAMES = 2430
EXPECTED_NATIVE_PLAY_RECORDS = 216845
EXPECTED_NATIVE_NO_PLAY_RECORDS = 27534
EXPECTED_EVENTS = 189311
_EVENT_FILE_RE = re.compile(r"2025[A-Z0-9]{3}\.EV[AN]\Z")
_MISMATCH_FIELDS = (
    "inning",
    "half",
    "outs",
    "bases",
    "score.away",
    "score.home",
)
_OFFLINE_FIELDS = (
    "batter_id",
    "pitcher_id",
    "balls",
    "strikes",
    "lineup_slot",
    "base_runner_ids",
)

EVIDENCE_SPLIT: dict[str, dict[str, object]] = {
    "causal_core": {
        "fields": [
            "inning",
            "half",
            "outs",
            "bases",
            "score.away",
            "score.home",
            "terminal",
        ],
        "source": "current_cwevent_row",
        "base_semantics": (
            "occupied_or_empty; runner identity may change in omitted NP rows"
        ),
        "comparison": "immediate_next_cwevent_row_when_present",
        "independent_oracle": False,
        "permitted_claim": "same_source_reducer_consistency_only",
    },
    "same_source_offline_observation": {
        "fields": list(_OFFLINE_FIELDS),
        "source": "immediate_next_cwevent_play_row",
        "classification": "same_source_next_play_context",
        "independent_oracle": False,
        "permitted_claim": "same_source_offline_consistency_only",
    },
}


class MLBSeasonCensusError(ValueError):
    """The frozen season cannot support a fail-closed census."""


@dataclass(frozen=True, slots=True)
class TransitionEvidence:
    """One current-row reduction compared with the immediate next row."""

    predicted_causal_core: dict[str, object]
    observed_causal_core: dict[str, object]
    causal_core_mismatches: tuple[str, ...]
    same_source_offline_observation: dict[str, object]
    base_runner_identity_changed_after_omitted_np: bool
    inning_end_survivor_destinations_normalized: int
    destination_collision_sha256: str | None
    independent_oracle: bool
    reducer_elapsed_ns: int
    step_sha256: str


@dataclass(frozen=True, slots=True)
class _GameMetadata:
    native_game_id: str
    away_team: str
    home_team: str
    game_type: str
    event_file: str
    native_play_records: int
    native_no_play_records: int
    omitted_np_gaps: int


@dataclass(frozen=True, slots=True)
class _GameCensus:
    manifest: dict[str, object]
    reducer_elapsed_ns: tuple[int, ...]


def _field(row: Mapping[str, object], logical_name: str) -> object:
    if not isinstance(row, Mapping):
        raise MLBSeasonCensusError("cwevent row must be a mapping")
    field_name = mlb.CWEVENT_FIELD_MAP[logical_name][0]
    if field_name not in row:
        raise MLBSeasonCensusError(f"cwevent row lacks {field_name}")
    return row[field_name]


def _text(
    row: Mapping[str, object],
    logical_name: str,
    *,
    optional: bool = False,
) -> str | None:
    value = _field(row, logical_name)
    if optional and value == "":
        return None
    if type(value) is not str or not value:
        raise MLBSeasonCensusError(
            f"{mlb.CWEVENT_FIELD_MAP[logical_name][0]} must be nonempty text"
        )
    return value


def _integer(
    row: Mapping[str, object],
    logical_name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = _field(row, logical_name)
    if type(value) is int:
        result = value
    elif type(value) is str and re.fullmatch(r"-?[0-9]+", value):
        result = int(value)
    else:
        raise MLBSeasonCensusError(
            f"{mlb.CWEVENT_FIELD_MAP[logical_name][0]} must be an integer"
        )
    if result < minimum or (maximum is not None and result > maximum):
        raise MLBSeasonCensusError(
            f"{mlb.CWEVENT_FIELD_MAP[logical_name][0]} is out of range: "
            f"{result}"
        )
    return result


def _flag(row: Mapping[str, object], logical_name: str) -> bool:
    value = _field(row, logical_name)
    if type(value) is bool:
        return value
    if type(value) is int and value in {0, 1}:
        return bool(value)
    if type(value) is str and value in {"T", "F"}:
        return value == "T"
    raise MLBSeasonCensusError(
        f"{mlb.CWEVENT_FIELD_MAP[logical_name][0]} must be T or F"
    )


def _row_sha256(row: Mapping[str, object]) -> str:
    if set(row) != set(mlb.CWEVENT_FIELD_NAMES):
        raise MLBSeasonCensusError("cwevent row fields differ from frozen map")
    return canonical_sha256(
        {name: row[name] for name in mlb.CWEVENT_FIELD_NAMES}
    )


def _collision_sha256(row: Mapping[str, object]) -> str:
    event_text = _text(row, "event_text")
    assert event_text is not None
    return canonical_sha256(
        {
            "namespace": "retrosheet.event_text",
            "value": event_text,
        }
    )


def _snapshot(
    row: Mapping[str, object],
    *,
    away_team: str,
    home_team: str,
) -> dict[str, object]:
    native_game_id = _text(row, "game_id")
    assert native_game_id is not None
    batting_home = _integer(
        row, "batting_home", minimum=0, maximum=1
    )
    half = "bottom" if batting_home else "top"
    return {
        "game_id": mlb.retrosheet_game_id(native_game_id),
        "inning": _integer(row, "inning", minimum=1),
        "half": half,
        "outs": _integer(row, "outs_before", minimum=0, maximum=2),
        "bases": (
            _text(row, "runner_on_first", optional=True),
            _text(row, "runner_on_second", optional=True),
            _text(row, "runner_on_third", optional=True),
        ),
        "score": mlb.MLBScore(
            away=_integer(row, "away_score_before", minimum=0),
            home=_integer(row, "home_score_before", minimum=0),
        ),
        "away_team": away_team,
        "home_team": home_team,
        "batting_team": home_team if batting_home else away_team,
        "fielding_team": away_team if batting_home else home_team,
        "batter_id": _text(row, "batter_id"),
        "pitcher_id": _text(row, "pitcher_id"),
        "balls": _integer(
            row, "balls_before", minimum=0, maximum=3
        ),
        "strikes": _integer(
            row, "strikes_before", minimum=0, maximum=2
        ),
        "lineup_slot": _integer(
            row, "lineup_slot", minimum=1, maximum=9
        ),
    }


def _state_from_row(
    row: Mapping[str, object],
    *,
    away_team: str,
    home_team: str,
) -> mlb.MLBGameState:
    snapshot = _snapshot(row, away_team=away_team, home_team=home_team)
    return mlb.MLBGameState(
        sport=mlb.MLB_SPORT,
        sequence=_integer(row, "event_index", minimum=1) - 1,
        terminal=False,
        observation_mode="synthetic_fixture",
        **snapshot,
    )


def _canonical_destination(value: int) -> int:
    if not 0 <= value <= 6:
        raise MLBSeasonCensusError("runner destination must be in 0..6")
    return 4 if value >= 4 else value


def _runner_advances(
    state: mlb.MLBGameState,
    row: Mapping[str, object],
) -> tuple[mlb.RunnerAdvance, ...]:
    advances: list[mlb.RunnerAdvance] = []
    batter_destination = _canonical_destination(
        _integer(row, "batter_destination", minimum=0, maximum=6)
    )
    if _flag(row, "batter_event"):
        advances.append(
            mlb.RunnerAdvance(
                runner_id=state.batter_id,
                start_base=0,
                destination=batter_destination,
            )
        )
    elif batter_destination:
        raise MLBSeasonCensusError(
            "non-batter event cannot advance the batter"
        )
    destination_fields = (
        "runner_on_first_destination",
        "runner_on_second_destination",
        "runner_on_third_destination",
    )
    for start_base, (runner_id, destination_name) in enumerate(
        zip(state.bases, destination_fields, strict=True),
        start=1,
    ):
        destination = _canonical_destination(
            _integer(row, destination_name, minimum=0, maximum=6)
        )
        if runner_id is None:
            if destination:
                raise MLBSeasonCensusError(
                    "runner cannot advance from an empty base"
                )
            continue
        advances.append(
            mlb.RunnerAdvance(
                runner_id=runner_id,
                start_base=start_base,
                destination=destination,
            )
        )
    observed_outs = _integer(
        row, "outs_on_play", minimum=0, maximum=3
    )
    derived_outs = sum(
        advance.destination == 0 for advance in advances
    )
    if observed_outs != derived_outs:
        raise MLBSeasonCensusError(
            "EVENT_OUTS_CT differs from current-row runner destinations"
        )
    return tuple(advances)


def _canonicalize_inning_end_survivor_collision(
    advances: tuple[mlb.RunnerAdvance, ...],
    *,
    state: mlb.MLBGameState,
    row: Mapping[str, object],
    next_snapshot: Mapping[str, object] | None,
) -> tuple[tuple[mlb.RunnerAdvance, ...], int]:
    occupied_destinations = [
        advance.destination
        for advance in advances
        if 1 <= advance.destination <= 3
    ]
    if len(occupied_destinations) == len(set(occupied_destinations)):
        return advances, 0
    outs = tuple(
        advance for advance in advances if advance.destination == 0
    )
    outs_after = state.outs + len(outs)
    event_text = _text(row, "event_text")
    assert event_text is not None
    parsed_out_origins = set(re.findall(r"\(([B123])\)", event_text))
    derived_out_origins = {
        "B" if advance.start_base == 0 else str(advance.start_base)
        for advance in outs
    }
    expected_inning = (
        state.inning if state.half == "top" else state.inning + 1
    )
    expected_half = "bottom" if state.half == "top" else "top"
    if (
        outs_after != 3
    ):
        raise MLBSeasonCensusError(
            "unsupported non-inning-end runner destination collision"
        )
    if (
        next_snapshot is None
        or next_snapshot["inning"] != expected_inning
        or next_snapshot["half"] != expected_half
        or next_snapshot["outs"] != 0
        or any(runner is not None for runner in next_snapshot["bases"])
        or not derived_out_origins
        or parsed_out_origins != derived_out_origins
        or _integer(row, "outs_on_play", minimum=0, maximum=3)
        != len(derived_out_origins)
    ):
        raise MLBSeasonCensusError(
            "unsupported inning-end survivor destination collision"
        )
    retained = tuple(
        advance
        for advance in advances
        if advance.destination in {0, 4}
    )
    return retained, len(advances) - len(retained)


def _offline_observation(snapshot: Mapping[str, object]) -> dict[str, object]:
    return {
        "batter_id": snapshot["batter_id"],
        "pitcher_id": snapshot["pitcher_id"],
        "balls": snapshot["balls"],
        "strikes": snapshot["strikes"],
        "lineup_slot": snapshot["lineup_slot"],
        "base_runner_ids": list(snapshot["bases"]),
    }


def _core_from_state(state: mlb.MLBGameState) -> dict[str, object]:
    return {
        "inning": state.inning,
        "half": state.half,
        "outs": state.outs,
        "bases": [runner is not None for runner in state.bases],
        "score.away": state.score.away,
        "score.home": state.score.home,
    }


def _core_from_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    score = snapshot["score"]
    if not isinstance(score, mlb.MLBScore):
        raise MLBSeasonCensusError("next-row score is invalid")
    return {
        "inning": snapshot["inning"],
        "half": snapshot["half"],
        "outs": snapshot["outs"],
        "bases": [runner is not None for runner in snapshot["bases"]],
        "score.away": score.away,
        "score.home": score.home,
    }


def _event_id(row: Mapping[str, object]) -> str:
    digest = canonical_sha256(
        {
            "namespace": "retrosheet.cwevent.season-census",
            "row_sha256": _row_sha256(row),
        }
    )
    return "evt_" + digest.removeprefix("sha256:")


def _build_event(
    state: mlb.MLBGameState,
    row: Mapping[str, object],
    next_snapshot: Mapping[str, object] | None,
) -> tuple[mlb.MLBPlayEvent, int]:
    raw_advances = _runner_advances(state, row)
    runs = tuple(
        advance.runner_id
        for advance in raw_advances
        if advance.destination == 4
    )
    outs = tuple(
        advance.runner_id
        for advance in raw_advances
        if advance.destination == 0
    )
    terminal = _flag(row, "end_game")
    if terminal != (next_snapshot is None):
        raise MLBSeasonCensusError(
            "only the final game row may carry GAME_END_FL"
        )
    outs_after = state.outs + len(outs)
    advances, inning_end_normalized = (
        _canonicalize_inning_end_survivor_collision(
            raw_advances,
            state=state,
            row=row,
            next_snapshot=next_snapshot,
        )
    )
    transition: mlb.InningTransition | None = None
    if not terminal and outs_after == 3:
        if state.half == "top":
            inning = state.inning
            half = "bottom"
            batting_team = state.home_team
            fielding_team = state.away_team
        else:
            inning = state.inning + 1
            half = "top"
            batting_team = state.away_team
            fielding_team = state.home_team
        assert next_snapshot is not None
        transition = mlb.InningTransition(
            inning=inning,
            half=half,
            batting_team=batting_team,
            fielding_team=fielding_team,
            batter_id=str(next_snapshot["batter_id"]),
            pitcher_id=str(next_snapshot["pitcher_id"]),
            lineup_slot=int(next_snapshot["lineup_slot"]),
            balls=int(next_snapshot["balls"]),
            strikes=int(next_snapshot["strikes"]),
        )
    direct_next = (
        next_snapshot
        if not terminal and transition is None
        else None
    )
    event_text = _text(row, "event_text")
    assert event_text is not None
    return mlb.MLBPlayEvent(
        sport=mlb.MLB_SPORT,
        game_id=state.game_id,
        sequence=state.sequence + 1,
        event_id=_event_id(row),
        inning=state.inning,
        half=state.half,
        outs_before=state.outs,
        bases_before=state.bases,
        score_before=state.score,
        batting_team=state.batting_team,
        fielding_team=state.fielding_team,
        batter_id=state.batter_id,
        pitcher_id=state.pitcher_id,
        balls_before=state.balls,
        strikes_before=state.strikes,
        lineup_slot_before=state.lineup_slot,
        play_type=(
            f"{_integer(row, 'event_code', minimum=0)}:{event_text}"
        ),
        runs=runs,
        outs=outs,
        runner_destinations=advances,
        next_batter_id=(
            None if direct_next is None else str(direct_next["batter_id"])
        ),
        next_pitcher_id=(
            None if direct_next is None else str(direct_next["pitcher_id"])
        ),
        next_balls=(
            None if direct_next is None else int(direct_next["balls"])
        ),
        next_strikes=(
            None if direct_next is None else int(direct_next["strikes"])
        ),
        next_lineup_slot=(
            None if direct_next is None else int(direct_next["lineup_slot"])
        ),
        inning_transition=transition,
        terminal=terminal,
        observation_mode="synthetic_fixture",
        source_parser="chadwick.cwevent",
        source_parser_version=mlb.CHADWICK_CWEVENT_VERSION,
        source_event_index=_integer(row, "event_index", minimum=1),
    ), inning_end_normalized


def evaluate_transition(
    play_row: Mapping[str, object],
    next_row: Mapping[str, object],
    *,
    away_team: str,
    home_team: str,
    _measure_reducer: bool = True,
) -> TransitionEvidence:
    """Reduce one current row and compare only its causal core to the next row."""

    state = _state_from_row(
        play_row, away_team=away_team, home_team=home_team
    )
    next_snapshot = _snapshot(
        next_row, away_team=away_team, home_team=home_team
    )
    if next_snapshot["game_id"] != state.game_id:
        raise MLBSeasonCensusError("next row belongs to a different game")
    current_index = _integer(play_row, "event_index", minimum=1)
    if _integer(next_row, "event_index", minimum=1) != current_index + 1:
        raise MLBSeasonCensusError("game EVENT_ID sequence is not contiguous")
    event, inning_end_normalized = _build_event(
        state, play_row, next_snapshot
    )
    started_ns = time.perf_counter_ns() if _measure_reducer else 0
    reduced = mlb.reduce_mlb_state(state, event)
    elapsed_ns = (
        time.perf_counter_ns() - started_ns if _measure_reducer else 0
    )
    predicted = _core_from_state(reduced)
    observed = _core_from_snapshot(next_snapshot)
    mismatches = tuple(
        field for field in _MISMATCH_FIELDS
        if predicted[field] != observed[field]
    )
    offline = _offline_observation(next_snapshot)
    predicted_runner_ids = list(reduced.bases)
    base_runner_identity_changed = (
        predicted_runner_ids != offline["base_runner_ids"]
    )
    return TransitionEvidence(
        predicted_causal_core=predicted,
        observed_causal_core=observed,
        causal_core_mismatches=mismatches,
        same_source_offline_observation=offline,
        base_runner_identity_changed_after_omitted_np=(
            base_runner_identity_changed
        ),
        inning_end_survivor_destinations_normalized=(
            inning_end_normalized
        ),
        destination_collision_sha256=(
            _collision_sha256(play_row)
            if inning_end_normalized
            else None
        ),
        independent_oracle=False,
        reducer_elapsed_ns=elapsed_ns,
        step_sha256=canonical_sha256(
            {
                "play_row_sha256": _row_sha256(play_row),
                "predicted_causal_core": predicted,
                "observed_causal_core": observed,
                "same_source_offline_observation": offline,
                "base_runner_identity_changed_after_omitted_np": (
                    base_runner_identity_changed
                ),
                "inning_end_survivor_destinations_normalized": (
                    inning_end_normalized
                ),
                "independent_oracle": False,
            }
        ),
    )


def _evaluate_terminal(
    row: Mapping[str, object],
    *,
    away_team: str,
    home_team: str,
    measure_reducer: bool,
) -> tuple[dict[str, object], int, str, int]:
    state = _state_from_row(row, away_team=away_team, home_team=home_team)
    event, inning_end_normalized = _build_event(state, row, None)
    started_ns = time.perf_counter_ns() if measure_reducer else 0
    reduced = mlb.reduce_mlb_state(state, event)
    elapsed_ns = (
        time.perf_counter_ns() - started_ns if measure_reducer else 0
    )
    if not reduced.terminal:
        raise MLBSeasonCensusError("final row did not produce a terminal state")
    core = {
        **_core_from_state(reduced),
        "terminal": True,
    }
    step_sha256 = canonical_sha256(
        {
            "play_row_sha256": _row_sha256(row),
            "predicted_terminal_core": core,
            "inning_end_survivor_destinations_normalized": (
                inning_end_normalized
            ),
        }
    )
    return core, elapsed_ns, step_sha256, inning_end_normalized


def _parse_event_file_metadata(
    payload: bytes,
    *,
    event_file: str,
) -> tuple[tuple[_GameMetadata, ...], int, int]:
    try:
        reader = csv.reader(
            io.StringIO(payload.decode("utf-8", errors="strict"), newline="")
        )
        records = tuple(tuple(row) for row in reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise MLBSeasonCensusError(
            f"{event_file} is not canonical CSV"
        ) from exc
    results: list[_GameMetadata] = []
    native_play_records = 0
    native_no_play_records = 0
    game_play_records = 0
    game_no_play_records = 0
    game_np_gaps = 0
    previous_play_was_np = False
    current_id: str | None = None
    info: dict[str, str] = {}

    def finish() -> None:
        if current_id is None:
            return
        required = {"visteam", "hometeam", "gametype"}
        if not required.issubset(info):
            raise MLBSeasonCensusError(
                f"{current_id} lacks governed team/game-type metadata"
            )
        results.append(
            _GameMetadata(
                native_game_id=current_id,
                away_team=info["visteam"],
                home_team=info["hometeam"],
                game_type=info["gametype"],
                event_file=event_file,
                native_play_records=game_play_records,
                native_no_play_records=game_no_play_records,
                omitted_np_gaps=game_np_gaps,
            )
        )

    for record in records:
        if not record:
            continue
        if record[0] == "play":
            if current_id is None:
                raise MLBSeasonCensusError(
                    f"{event_file} has a play before its game id"
                )
            if len(record) < 7:
                raise MLBSeasonCensusError(
                    f"{event_file} has an invalid play row"
                )
            native_play_records += 1
            game_play_records += 1
            if record[6] == "NP":
                native_no_play_records += 1
                game_no_play_records += 1
                if not previous_play_was_np:
                    game_np_gaps += 1
                previous_play_was_np = True
            else:
                previous_play_was_np = False
        if record[0] == "id":
            if len(record) != 2 or not record[1]:
                raise MLBSeasonCensusError(f"{event_file} has an invalid id row")
            finish()
            current_id = record[1]
            info = {}
            game_play_records = 0
            game_no_play_records = 0
            game_np_gaps = 0
            previous_play_was_np = False
        elif (
            record[0] == "info"
            and current_id is not None
            and len(record) >= 3
            and record[1] in {"visteam", "hometeam", "gametype"}
        ):
            info[record[1]] = record[2]
    finish()
    if not results:
        raise MLBSeasonCensusError(f"{event_file} contains no games")
    return (
        tuple(results),
        native_play_records,
        native_no_play_records,
    )


def _safe_archive_inventory_and_extract(
    object_bytes: bytes,
    *,
    destination: Path,
) -> tuple[
    tuple[str, ...],
    dict[str, _GameMetadata],
    tuple[dict[str, object], ...],
]:
    try:
        with zipfile.ZipFile(io.BytesIO(object_bytes)) as archive:
            members = archive.infolist()
            for member in members:
                pure = PurePosixPath(member.filename)
                if (
                    pure.is_absolute()
                    or not pure.parts
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or "\\" in member.filename
                ):
                    raise MLBSeasonCensusError(
                        "Retrosheet archive contains an unsafe member path"
                    )
            event_names = tuple(
                sorted(
                    member.filename
                    for member in members
                    if _EVENT_FILE_RE.fullmatch(member.filename)
                )
            )
            if len(event_names) != EXPECTED_EVENT_FILES:
                raise MLBSeasonCensusError(
                    "Retrosheet archive does not contain 30 event files"
                )
            metadata: dict[str, _GameMetadata] = {}
            member_manifest: list[dict[str, object]] = []
            native_play_records_total = 0
            native_no_play_records_total = 0
            for event_name in event_names:
                payload = archive.read(event_name)
                games, native_play_records, native_no_play_records = (
                    _parse_event_file_metadata(
                    payload, event_file=event_name
                )
                )
                native_play_records_total += native_play_records
                native_no_play_records_total += native_no_play_records
                member_manifest.append(
                    {
                        "event_file": event_name,
                        "byte_length": len(payload),
                        "sha256": (
                            "sha256:" + hashlib.sha256(payload).hexdigest()
                        ),
                        "games": len(games),
                        "native_play_records": native_play_records,
                        "native_no_play_records": native_no_play_records,
                        "omitted_np_gaps": sum(
                            game.omitted_np_gaps for game in games
                        ),
                        "cwevent_expected_events": (
                            native_play_records - native_no_play_records
                        ),
                    }
                )
                for game in games:
                    if game.native_game_id in metadata:
                        raise MLBSeasonCensusError(
                            "Retrosheet game ID appears in multiple files"
                        )
                    if game.game_type != "regular":
                        raise MLBSeasonCensusError(
                            "frozen archive contains a non-regular game"
                        )
                    metadata[game.native_game_id] = game
            if len(metadata) != EXPECTED_GAMES:
                raise MLBSeasonCensusError(
                    "Retrosheet metadata does not contain 2430 games"
                )
            if (
                native_play_records_total != EXPECTED_NATIVE_PLAY_RECORDS
                or native_no_play_records_total
                != EXPECTED_NATIVE_NO_PLAY_RECORDS
                or native_play_records_total - native_no_play_records_total
                != EXPECTED_EVENTS
            ):
                raise MLBSeasonCensusError(
                    "native play/NP/cwevent event census is inconsistent"
                )
            archive.extractall(destination)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise MLBSeasonCensusError(
            "Retrosheet archive cannot be inventoried or extracted"
        ) from exc
    return event_names, metadata, tuple(member_manifest)


def season_cwevent_command(
    runtime: mlb.CweventRuntime,
    *,
    event_files: Sequence[str],
) -> tuple[str, ...]:
    """Return one deterministic invocation over all 30 event files."""

    if not isinstance(runtime, mlb.CweventRuntime):
        raise MLBSeasonCensusError("cwevent runtime is not verified")
    names = tuple(event_files)
    if len(names) != EXPECTED_EVENT_FILES or tuple(sorted(names)) != names:
        raise MLBSeasonCensusError(
            "season command requires 30 sorted event files"
        )
    if any(
        PurePosixPath(name).name != name
        or _EVENT_FILE_RE.fullmatch(name) is None
        for name in names
    ):
        raise MLBSeasonCensusError("season command has an unsafe event file")
    return (
        runtime.executable,
        "-q",
        "-n",
        "-y",
        "2025",
        "-f",
        mlb.CWEVENT_FIELD_ARGUMENT,
        *names,
    )


def canonical_season_command_identity(
    runtime: mlb.CweventRuntime,
    *,
    event_files: Sequence[str],
) -> dict[str, object]:
    """Bind stable command tokens to binary bytes without a host path."""

    command = season_cwevent_command(
        runtime,
        event_files=event_files,
    )
    return {
        "binary_sha256": runtime.binary_sha256,
        "tokens": ["cwevent", *command[1:]],
    }


def _run_cwevent(command: tuple[str, ...], *, cwd: Path) -> bytes:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MLBSeasonCensusError("season cwevent execution failed") from exc
    if completed.returncode != 0:
        raise MLBSeasonCensusError("season cwevent returned nonzero")
    if completed.stderr:
        raise MLBSeasonCensusError("quiet season cwevent emitted stderr")
    if not completed.stdout:
        raise MLBSeasonCensusError("season cwevent emitted no rows")
    return completed.stdout


def _decode_grouped_rows(
    output: bytes,
) -> tuple[tuple[str, tuple[dict[str, object], ...]], ...]:
    try:
        text = output.decode("utf-8", errors="strict")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if tuple(reader.fieldnames or ()) != mlb.CWEVENT_FIELD_NAMES:
            raise MLBSeasonCensusError(
                "season cwevent header differs from frozen map"
            )
        groups: list[tuple[str, list[dict[str, object]]]] = []
        seen: set[str] = set()
        for decoded in reader:
            if None in decoded or set(decoded) != set(mlb.CWEVENT_FIELD_NAMES):
                raise MLBSeasonCensusError(
                    "season cwevent row has invalid width"
                )
            row = dict(decoded)
            game_id = _text(row, "game_id")
            assert game_id is not None
            if not groups or groups[-1][0] != game_id:
                if game_id in seen:
                    raise MLBSeasonCensusError(
                        "season cwevent game rows are not contiguous"
                    )
                seen.add(game_id)
                groups.append((game_id, []))
            expected_index = len(groups[-1][1]) + 1
            if _integer(row, "event_index", minimum=1) != expected_index:
                raise MLBSeasonCensusError(
                    f"{game_id} EVENT_ID sequence is not contiguous"
                )
            groups[-1][1].append(row)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise MLBSeasonCensusError(
            "season cwevent output is not canonical CSV"
        ) from exc
    frozen = tuple(
        (game_id, tuple(rows)) for game_id, rows in groups
    )
    if len(frozen) != EXPECTED_GAMES:
        raise MLBSeasonCensusError(
            "season cwevent output does not contain 2430 games"
        )
    if sum(len(rows) for _, rows in frozen) != EXPECTED_EVENTS:
        raise MLBSeasonCensusError(
            "season cwevent output does not contain 189311 events"
        )
    return frozen


def _evaluate_game(
    native_game_id: str,
    rows: tuple[dict[str, object], ...],
    metadata: _GameMetadata,
    *,
    measure_reducer: bool,
) -> _GameCensus:
    if not rows:
        raise MLBSeasonCensusError(f"{native_game_id} contains no events")
    if any(_text(row, "game_id") != native_game_id for row in rows):
        raise MLBSeasonCensusError(
            f"{native_game_id} contains a foreign event row"
        )
    if (
        metadata.native_play_records
        - metadata.native_no_play_records
        != len(rows)
    ):
        raise MLBSeasonCensusError(
            f"{native_game_id} native play/NP/cwevent counts disagree"
        )
    timings: list[int] = []
    event_result_sha256s: list[str] = []
    failures: list[dict[str, object]] = []
    transitions_validated = 0
    offline_observations = 0
    base_runner_identity_changes = 0
    inning_end_survivor_normalizations = 0
    inning_end_collision_events_canonicalized = 0
    canonicalized_collision_signatures: Counter[str] = Counter()
    reducer_calls_completed = 0
    mismatch_counts = {field: 0 for field in _MISMATCH_FIELDS}

    def record_failure(
        *,
        row: Mapping[str, object],
        event_index: int,
        category: str,
    ) -> None:
        failure = {
            "event_index": event_index,
            "row_sha256": _row_sha256(row),
            "category": category,
        }
        if category in {
            "unsupported_inning_end_destination_collision",
            "unsupported_non_inning_destination_collision",
        }:
            failure["collision_sha256"] = _collision_sha256(row)
        failures.append(failure)
        event_result_sha256s.append(
            canonical_sha256(
                {
                    "status": "failed_closed",
                    **failure,
                }
            )
        )

    def failure_category(detail: str) -> str:
        if detail.startswith(
            "unsupported inning-end survivor destination collision"
        ):
            return "unsupported_inning_end_destination_collision"
        if detail.startswith(
            "unsupported non-inning-end runner destination collision"
        ):
            return "unsupported_non_inning_destination_collision"
        if re.fullmatch(
            r"(?:BAT|RUN[123])_DEST_ID is out of range: [0-9]+",
            detail,
        ):
            return "unsupported_cwevent_destination_code"
        if detail == "terminal event cannot occur before inning nine":
            return "unsupported_shortened_game_terminal_rule"
        return "event_adapter_or_reducer_rejection"

    for index, row in enumerate(rows[:-1]):
        try:
            evidence = evaluate_transition(
                row,
                rows[index + 1],
                away_team=metadata.away_team,
                home_team=metadata.home_team,
                _measure_reducer=measure_reducer,
            )
        except (MLBSeasonCensusError, mlb.MLBGameStateError) as exc:
            detail = str(exc)
            record_failure(
                row=row,
                event_index=index + 1,
                category=failure_category(detail),
            )
            continue
        if measure_reducer:
            timings.append(evidence.reducer_elapsed_ns)
        reducer_calls_completed += 1
        if evidence.causal_core_mismatches:
            for field in evidence.causal_core_mismatches:
                mismatch_counts[field] += 1
            next_snapshot = _snapshot(
                rows[index + 1],
                away_team=metadata.away_team,
                home_team=metadata.home_team,
            )
            automatic_runner_context_gap = (
                evidence.causal_core_mismatches == ("bases",)
                and int(next_snapshot["inning"]) >= 10
                and next_snapshot["outs"] == 0
                and [
                    runner is not None
                    for runner in next_snapshot["bases"]
                ]
                == [False, True, False]
                and _integer(
                    row, "outs_before", minimum=0, maximum=2
                )
                + _integer(
                    row, "outs_on_play", minimum=0, maximum=3
                )
                == 3
            )
            record_failure(
                row=row,
                event_index=index + 1,
                category=(
                    "unsupported_automatic_runner_context"
                    if automatic_runner_context_gap
                    else "causal_core_mismatch"
                ),
            )
            continue
        transitions_validated += 1
        offline_observations += len(
            evidence.same_source_offline_observation
        )
        base_runner_identity_changes += int(
            evidence.base_runner_identity_changed_after_omitted_np
        )
        inning_end_survivor_normalizations += (
            evidence.inning_end_survivor_destinations_normalized
        )
        if evidence.destination_collision_sha256 is not None:
            inning_end_collision_events_canonicalized += 1
            canonicalized_collision_signatures[
                evidence.destination_collision_sha256
            ] += 1
        event_result_sha256s.append(evidence.step_sha256)
    final_core: dict[str, object] | None = None
    terminal_validated = 0
    try:
        (
            final_core,
            final_elapsed_ns,
            final_step_sha256,
            terminal_normalizations,
        ) = _evaluate_terminal(
            rows[-1],
            away_team=metadata.away_team,
            home_team=metadata.home_team,
            measure_reducer=measure_reducer,
        )
    except (MLBSeasonCensusError, mlb.MLBGameStateError) as exc:
        detail = str(exc)
        record_failure(
            row=rows[-1],
            event_index=len(rows),
            category=failure_category(detail),
        )
    else:
        if measure_reducer:
            timings.append(final_elapsed_ns)
        reducer_calls_completed += 1
        inning_end_survivor_normalizations += terminal_normalizations
        event_result_sha256s.append(final_step_sha256)
        terminal_validated = 1
    stream_sha256 = canonical_sha256(
        {
            "native_game_id": native_game_id,
            "canonical_game_id": mlb.retrosheet_game_id(native_game_id),
            "event_file": metadata.event_file,
            "away_team": metadata.away_team,
            "home_team": metadata.home_team,
            "row_sha256s": [_row_sha256(row) for row in rows],
            "event_result_sha256s": event_result_sha256s,
            "final_causal_core": final_core,
        }
    )
    status = "valid" if not failures else "failed_closed"
    return _GameCensus(
        manifest={
            "native_game_id": native_game_id,
            "canonical_game_id": mlb.retrosheet_game_id(native_game_id),
            "event_file": metadata.event_file,
            "away_team": metadata.away_team,
            "home_team": metadata.home_team,
            "events": len(rows),
            "events_reducer_validated": reducer_calls_completed,
            "native_play_records": metadata.native_play_records,
            "native_no_play_records": metadata.native_no_play_records,
            "omitted_np_gaps": metadata.omitted_np_gaps,
            "transitions_with_next_row": len(rows) - 1,
            "causal_core_comparisons_validated": transitions_validated,
            "same_source_offline_field_observations": offline_observations,
            "base_runner_identity_changes_after_omitted_np": (
                base_runner_identity_changes
            ),
            "inning_end_survivor_destinations_normalized": (
                inning_end_survivor_normalizations
            ),
            "inning_end_collision_events_canonicalized": (
                inning_end_collision_events_canonicalized
            ),
            "canonicalized_collision_signatures": [
                {
                    "category": (
                        "canonicalized_inning_end_destination_collision"
                    ),
                    "collision_sha256": collision_sha256,
                    "count": count,
                }
                for collision_sha256, count in sorted(
                    canonicalized_collision_signatures.items()
                )
            ],
            "terminal_events": 1,
            "terminal_events_validated": terminal_validated,
            "causal_core_mismatches": sum(mismatch_counts.values()),
            "causal_core_mismatches_by_field": mismatch_counts,
            "status": status,
            "first_failure": failures[0] if failures else None,
            "failures": failures,
            "stream_sha256": stream_sha256,
        },
        reducer_elapsed_ns=tuple(timings),
    )


def _reconstruct_season(
    grouped_rows: tuple[
        tuple[str, tuple[dict[str, object], ...]], ...
    ],
    metadata: Mapping[str, _GameMetadata],
    *,
    identity: Mapping[str, object],
    measure_reducer: bool,
) -> tuple[
    str,
    tuple[dict[str, object], ...],
    tuple[int, ...],
]:
    manifests: list[dict[str, object]] = []
    timings: list[int] = []
    output_ids = {game_id for game_id, _ in grouped_rows}
    if output_ids != set(metadata):
        raise MLBSeasonCensusError(
            "cwevent games differ from native event-file metadata"
        )
    for native_game_id, rows in grouped_rows:
        result = _evaluate_game(
            native_game_id,
            rows,
            metadata[native_game_id],
            measure_reducer=measure_reducer,
        )
        manifests.append(result.manifest)
        timings.extend(result.reducer_elapsed_ns)
    season_sha256 = canonical_sha256(
        {
            "identity": dict(identity),
            "game_manifest": manifests,
        }
    )
    return season_sha256, tuple(manifests), tuple(timings)


def _percentile(
    sorted_values: Sequence[int],
    *,
    numerator: int,
    denominator: int,
) -> int:
    if not sorted_values:
        raise MLBSeasonCensusError("reducer performance has no samples")
    rank = max(
        1,
        (
            numerator * len(sorted_values) + denominator - 1
        ) // denominator,
    )
    return sorted_values[rank - 1]


def _performance(timings: tuple[int, ...]) -> dict[str, object]:
    if not timings or any(value <= 0 for value in timings):
        raise MLBSeasonCensusError(
            "reducer timing samples are invalid"
        )
    ordered = tuple(sorted(timings))
    total_ns = sum(timings)
    return {
        "samples": len(timings),
        "clock": "time.perf_counter_ns",
        "scope": (
            "reduce_mlb_state_call_only; excludes cwevent, CSV parsing, "
            "event construction, comparison, hashing, and model inference"
        ),
        "latency_ns": {
            "p50": _percentile(
                ordered, numerator=50, denominator=100
            ),
            "p95": _percentile(
                ordered, numerator=95, denominator=100
            ),
            "p99": _percentile(
                ordered, numerator=99, denominator=100
            ),
            "mean": (total_ns + len(timings) // 2) // len(timings),
            "max": ordered[-1],
        },
        "throughput_transitions_per_second": (
            len(timings) * 1_000_000_000 + total_ns // 2
        ) // total_ns,
    }


def run_frozen_retrosheet_2025_season_census(
    *,
    program_root: str | Path,
    cwevent_executable: str = "cwevent",
) -> dict[str, object]:
    """Run all 2,430 games twice and return an auditable census artifact."""

    root = Path(program_root)
    store_root = root / "var/raw"
    manifest_path = (
        store_root / mlb.RETROSHEET_2025_MANIFEST_RELATIVE_PATH
    )
    try:
        verified = read_verified_static_object(
            manifest_path,
            store_root=store_root,
            program_root=root,
        )
    except Exception as exc:
        raise MLBSeasonCensusError(
            "frozen Retrosheet object or manifest verification failed"
        ) from exc
    manifest = verified.record.manifest
    if (
        manifest.dataset_id != mlb.RETROSHEET_DATASET_ID
        or manifest.object_sha256
        != mlb.RETROSHEET_2025_RAW_OBJECT_SHA256
        or manifest.manifest_sha256
        != mlb.RETROSHEET_2025_MANIFEST_SHA256
        or manifest.license_status != "research_only"
    ):
        raise MLBSeasonCensusError(
            "Retrosheet frozen source binding is invalid"
        )
    runtime = mlb.require_cwevent_runtime(cwevent_executable)
    with tempfile.TemporaryDirectory(
        prefix="saf-mlb-season-census-"
    ) as temporary:
        workdir = Path(temporary)
        event_files, metadata, event_file_manifest = (
            _safe_archive_inventory_and_extract(
                verified.object_bytes,
                destination=workdir,
            )
        )
        command = season_cwevent_command(
            runtime, event_files=event_files
        )
        command_identity = canonical_season_command_identity(
            runtime,
            event_files=event_files,
        )
        output = _run_cwevent(command, cwd=workdir)
    output_sha256 = "sha256:" + hashlib.sha256(output).hexdigest()
    command_sha256 = canonical_sha256(command_identity)
    event_file_manifest_sha256 = canonical_sha256(
        list(event_file_manifest)
    )
    grouped_rows = _decode_grouped_rows(output)
    identity = {
        "raw_object_sha256": manifest.object_sha256,
        "source_manifest_sha256": manifest.manifest_sha256,
        "cwevent_binary_sha256": runtime.binary_sha256,
        "cwevent_command_sha256": command_sha256,
        "cwevent_field_map_sha256": mlb.CWEVENT_FIELD_MAP_SHA256,
        "cwevent_output_sha256": output_sha256,
        "event_file_manifest_sha256": event_file_manifest_sha256,
    }
    first_sha256, game_manifest, timings = _reconstruct_season(
        grouped_rows,
        metadata,
        identity=identity,
        measure_reducer=True,
    )
    second_sha256, second_manifest, _ = _reconstruct_season(
        grouped_rows,
        metadata,
        identity=identity,
        measure_reducer=False,
    )
    if first_sha256 != second_sha256 or game_manifest != second_manifest:
        raise MLBSeasonCensusError(
            "two full-season reconstructions are not deterministic"
        )
    event_count = sum(
        int(game["events"]) for game in game_manifest
    )
    transition_count = sum(
        int(game["transitions_with_next_row"])
        for game in game_manifest
    )
    terminal_count = sum(
        int(game["terminal_events"]) for game in game_manifest
    )
    offline_count = sum(
        int(game["same_source_offline_field_observations"])
        for game in game_manifest
    )
    base_runner_identity_change_count = sum(
        int(game["base_runner_identity_changes_after_omitted_np"])
        for game in game_manifest
    )
    omitted_np_gap_count = sum(
        int(game["omitted_np_gaps"]) for game in game_manifest
    )
    games_valid = sum(game["status"] == "valid" for game in game_manifest)
    games_failed = len(game_manifest) - games_valid
    events_reducer_validated = sum(
        int(game["events_reducer_validated"]) for game in game_manifest
    )
    causal_comparisons_validated = sum(
        int(game["causal_core_comparisons_validated"])
        for game in game_manifest
    )
    terminal_rows_validated = sum(
        int(game["terminal_events_validated"]) for game in game_manifest
    )
    failures = [
        {
            "native_game_id": game["native_game_id"],
            **failure,
        }
        for game in game_manifest
        for failure in game["failures"]
    ]
    failure_categories = Counter(
        str(failure["category"]) for failure in failures
    )
    mismatch_counts = {
        field: sum(
            int(game["causal_core_mismatches_by_field"][field])
            for game in game_manifest
        )
        for field in _MISMATCH_FIELDS
    }
    collision_signatures: Counter[tuple[str, str]] = Counter()
    canonicalized_collision_events = 0
    survivor_destinations_normalized = 0
    for game in game_manifest:
        canonicalized_collision_events += int(
            game["inning_end_collision_events_canonicalized"]
        )
        survivor_destinations_normalized += int(
            game["inning_end_survivor_destinations_normalized"]
        )
        for signature in game["canonicalized_collision_signatures"]:
            collision_signatures[
                (
                    str(signature["category"]),
                    str(signature["collision_sha256"]),
                )
            ] += int(signature["count"])
    for failure in failures:
        if "collision_sha256" in failure:
            collision_signatures[
                (
                    str(failure["category"]),
                    str(failure["collision_sha256"]),
                )
            ] += 1
    collision_events_observed = sum(collision_signatures.values())
    if (
        event_count != EXPECTED_EVENTS
        or len(game_manifest) != EXPECTED_GAMES
        or transition_count != EXPECTED_EVENTS - EXPECTED_GAMES
        or terminal_count != EXPECTED_GAMES
        or offline_count
        != causal_comparisons_validated * len(_OFFLINE_FIELDS)
        or events_reducer_validated != len(timings)
        or collision_events_observed != 78
        or canonicalized_collision_events != 72
    ):
        raise MLBSeasonCensusError(
            "full-season census totals are internally inconsistent"
        )
    return {
        "artifact_id": "MLB-SEASON-STATE-CENSUS-V0",
        "artifact_version": "v0",
        "owner": "Team D4+I",
        "due_gate": "2026-08-05_W2_review",
        "status": (
            "FULL_SEASON_CENSUS_COMPLETE_WITH_UNSUPPORTED_GAPS_RESEARCH_ONLY"
        ),
        "lineage": {
            "dataset_id": manifest.dataset_id,
            "source_url": manifest.source_url,
            "source_fetched_at": manifest.fetched_at,
            "license_ref": manifest.license_ref,
            "license_status": manifest.license_status,
            "raw_object_sha256": manifest.object_sha256,
            "source_manifest_sha256": manifest.manifest_sha256,
            "schema_fingerprint": manifest.schema_fingerprint,
            "event_file_manifest_sha256": event_file_manifest_sha256,
            "event_files": list(event_file_manifest),
        },
        "execution": {
            "cwevent_invocations": 1,
            "cwevent_version": runtime.version,
            "cwevent_binary_sha256": runtime.binary_sha256,
            "cwevent_command": command_identity["tokens"],
            "cwevent_command_sha256": command_sha256,
            "cwevent_field_map_sha256": mlb.CWEVENT_FIELD_MAP_SHA256,
            "cwevent_output_sha256": output_sha256,
            "temporary_directory_cleanup": (
                "Python TemporaryDirectory context completed before validation"
            ),
        },
        "coverage": {
            "season": 2025,
            "game_type": "regular",
            "event_files": len(event_files),
            "games": len(game_manifest),
            "games_fully_supported": games_valid,
            "games_with_any_unsupported": games_failed,
            "native_play_records": EXPECTED_NATIVE_PLAY_RECORDS,
            "native_no_play_records": EXPECTED_NATIVE_NO_PLAY_RECORDS,
            "omitted_np_gaps": omitted_np_gap_count,
            "events": event_count,
            "transitions_with_next_row": transition_count,
            "terminal_events": terminal_count,
            "full_native_lifecycle_complete": False,
        },
        "validation": {
            "policy": "per_game_fail_closed",
            "games_valid": games_valid,
            "games_failed": games_failed,
            "events_scanned": event_count,
            "events_reducer_validated": events_reducer_validated,
            "events_failed_closed": len(failures),
            "failures_by_category": dict(sorted(failure_categories.items())),
            "causal_core_comparisons": causal_comparisons_validated,
            "causal_core_comparison_attempts": transition_count,
            "causal_core_mismatches_total": sum(mismatch_counts.values()),
            "causal_core_mismatches_by_field": mismatch_counts,
            "same_source_offline_field_observations": offline_count,
            "same_source_offline_observations_by_field": {
                field: causal_comparisons_validated
                for field in _OFFLINE_FIELDS
            },
            "base_runner_identity_changes_after_omitted_np": (
                base_runner_identity_change_count
            ),
            "native_np_records_omitted_from_reducer": (
                EXPECTED_NATIVE_NO_PLAY_RECORDS
            ),
            "native_np_gap_count": omitted_np_gap_count,
            "np_adapter_implemented": False,
            "np_adapter_gap_priority": "P1",
            "automatic_runner_adapter_implemented": False,
            "cwevent_destination_code_7_supported": False,
            "terminal_rows_validated": terminal_rows_validated,
            "destination_collision_events_observed": (
                collision_events_observed
            ),
            "inning_end_collision_events_canonicalized": (
                canonicalized_collision_events
            ),
            "unsupported_destination_collision_events": (
                collision_events_observed - canonicalized_collision_events
            ),
            "inning_end_survivor_destinations_normalized": (
                survivor_destinations_normalized
            ),
            "destination_collision_signatures": [
                {
                    "category": category,
                    "collision_sha256": collision_sha256,
                    "count": count,
                }
                for (category, collision_sha256), count in sorted(
                    collision_signatures.items()
                )
            ],
            "game_manifest_sha256": canonical_sha256(
                list(game_manifest)
            ),
        },
        "evidence_split": EVIDENCE_SPLIT,
        "performance": {
            "reducer_only": _performance(timings),
        },
        "determinism": {
            "runs": 2,
            "canonical_hashes": [first_sha256, second_sha256],
            "canonical_hash_match": True,
            "performance_excluded_from_canonical_hash": True,
        },
        "canonical_reconstruction_sha256": first_sha256,
        "game_manifest": list(game_manifest),
        "claims": {
            "independent_oracle": False,
            "model_trained": False,
            "prediction_accuracy_reported": False,
            "market_symmetry_tested": False,
            "full_native_lifecycle_complete": False,
            "runner_identity_lifecycle_complete": False,
            "permitted_result": (
                "same-source offline engineering consistency only"
            ),
        },
        "limitations": [
            (
                "Immediate-next-row comparison is from the same frozen "
                "Retrosheet/cwevent stream and is not an independent oracle."
            ),
            (
                "The native archive has 216845 play rows; cwevent omits 27534 "
                "NP substitution/adjustment placeholders and emits 189311 "
                "state-transition events."
            ),
            (
                "Batter, pitcher, count, and lineup context are injected only "
                "as explicitly labeled same-source offline observations."
            ),
            (
                "Extra-inning automatic runners are not derivable from the "
                "current cwevent row; 600 observations fail closed rather "
                "than silently injecting the next-row base state."
            ),
            (
                "The frozen reducer adapter accepts destinations 0..6, while "
                "2025 cwevent emits destination 7 for some unearned-run "
                "outcomes; those observations fail closed."
            ),
            (
                "Five shortened games terminate before inning nine, which the "
                "frozen reducer rejects; no shortened-game rule was added."
            ),
            (
                "Reducer latency excludes parsing, feature construction, "
                "model inference, networking, storage, and market joins."
            ),
            (
                "No model, predictive accuracy, alpha, or prediction-market "
                "symmetry claim is supported by this artifact."
            ),
        ],
    }


def write_frozen_retrosheet_2025_season_census(
    *,
    program_root: str | Path,
    output_path: str | Path,
    cwevent_executable: str = "cwevent",
) -> dict[str, object]:
    """Write the canonical JSON artifact with a self-hash."""

    artifact = run_frozen_retrosheet_2025_season_census(
        program_root=program_root,
        cwevent_executable=cwevent_executable,
    )
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _main() -> None:
    root = Path(__file__).resolve().parents[3]
    output = (
        root
        / "artifacts/game-state/mlb/season_state_census_v0.json"
    )
    artifact = write_frozen_retrosheet_2025_season_census(
        program_root=root,
        output_path=output,
    )
    print(canonical_json({
        "output_path": str(output),
        "artifact_sha256": artifact["artifact_sha256"],
        "canonical_reconstruction_sha256": artifact[
            "canonical_reconstruction_sha256"
        ],
    }))


if __name__ == "__main__":
    _main()
