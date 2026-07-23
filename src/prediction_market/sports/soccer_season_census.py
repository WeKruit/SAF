"""Reproducible reducer-only census over the frozen StatsBomb PL season."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from prediction_market.contracts import (
    canonical_sha256,
    event_id_for,
    payload_sha256,
)
from prediction_market.sports.soccer_game_state import (
    SoccerGameEvent,
    SoccerGameStateError,
    initial_soccer_game_state,
    reduce_soccer_game_state,
    statsbomb_event_payload,
)
from prediction_market.sports.statsbomb import (
    STATSBOMB_COMMIT,
    STATSBOMB_EXPECTED_MATCHES,
    inspect_statsbomb_event,
    inspect_statsbomb_match_index,
)
from prediction_market.static_store import read_verified_static_object


_DATASET_ID = "DS-STATSBOMB-OPEN"
_SOURCE = "statsbomb"
_COMPETITION_ID = "cmp_statsbomb_2"
_EXPERIMENT_ID = "X-12"
_COORDINATE_FLAG = "source_coordinate_out_of_bounds"


class SoccerSeasonCensusError(ValueError):
    """The frozen season cannot support a reproducible reducer census."""


@dataclass(frozen=True, slots=True)
class FrozenStatsBombEventPartition:
    """Verified native events and the immutable object binding that contains them."""

    match_id: int
    object_sha256: str
    fetched_at: str
    events: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class SoccerSeasonCensusFailure:
    match_id: int
    failed_sequence: int | None
    error: str


@dataclass(frozen=True, slots=True)
class SoccerSeasonCensusRun:
    games_total: int
    completed_games: int
    fail_closed_games: int
    source_events: int
    events_reduced: int
    score_mismatches: int
    score_mismatch_match_ids: tuple[int, ...]
    clock_regressions: int
    quality_flag_counts: tuple[tuple[str, int], ...]
    canonical_state_sha256: str
    failures: tuple[SoccerSeasonCensusFailure, ...]


@dataclass(frozen=True, slots=True)
class SoccerSeasonCensusReport:
    census_version: str
    dataset_id: str
    source_version: str
    scan_runs: int
    deterministic: bool
    canonical_state_sha256: str
    run_summaries: tuple[SoccerSeasonCensusRun, ...]

    def to_document(self) -> dict[str, object]:
        document = asdict(self)
        for summary in document["run_summaries"]:  # type: ignore[index]
            summary["quality_flag_counts"] = dict(  # type: ignore[index]
                summary["quality_flag_counts"]  # type: ignore[index]
            )
        return document


EventLoader = Callable[[int], FrozenStatsBombEventPartition]


def _required_match_int(
    match: Mapping[str, Any],
    field: str,
    *,
    minimum: int = 0,
) -> int:
    value = match.get(field)
    if type(value) is not int or value < minimum:
        raise SoccerSeasonCensusError(
            f"match {field} must be an integer >= {minimum}"
        )
    return value


def _team_id(match: Mapping[str, Any], side: str) -> int:
    team = match.get(f"{side}_team")
    if type(team) is not dict:
        raise SoccerSeasonCensusError(f"match {side}_team must be an object")
    value = team.get(f"{side}_team_id")
    if type(value) is not int or value <= 0:
        raise SoccerSeasonCensusError(
            f"match {side}_team_id must be a positive integer"
        )
    return value


def _location_is_out_of_bounds(value: object) -> bool:
    if type(value) is not list or len(value) not in {2, 3}:
        return False
    bounds = (Decimal("120"), Decimal("80"))
    for coordinate, maximum in zip(value[:2], bounds, strict=True):
        if type(coordinate) not in {int, float}:
            return False
        try:
            decimal = Decimal(str(coordinate))
        except InvalidOperation:
            return False
        if not decimal.is_finite():
            return False
        if decimal < 0 or decimal > maximum:
            return True
    return False


def _has_source_coordinate_out_of_bounds(
    raw_event: Mapping[str, Any],
) -> bool:
    locations: list[object] = [raw_event.get("location")]
    for action_field in ("carry", "goalkeeper", "pass", "shot"):
        action = raw_event.get(action_field)
        if type(action) is dict:
            locations.append(action.get("end_location"))
    return any(_location_is_out_of_bounds(location) for location in locations)


def _event_envelope_id(
    *,
    payload: Mapping[str, object],
    object_sha256: str,
    fetched_at: str,
    match_id: int,
    home_team_id: int,
    away_team_id: int,
    quality_flags: tuple[str, ...],
) -> str:
    sequence = payload["sequence"]
    native_event_id = payload["native_event_id"]
    if type(sequence) is not int or sequence <= 0:
        raise SoccerSeasonCensusError("normalized event sequence is invalid")
    if type(native_event_id) is not str or not native_event_id:
        raise SoccerSeasonCensusError("normalized native event id is invalid")
    game_id = f"game_statsbomb_{match_id}"
    canonical_refs = {
        "competition_id": _COMPETITION_ID,
        "game_id": game_id,
        "participant_ids": (
            f"participant_statsbomb_{home_team_id}",
            f"participant_statsbomb_{away_team_id}",
        ),
        "venue_event_id": None,
        "market_id": None,
        "outcome_id": None,
        "condition_id": None,
    }
    native_refs = (
        {
            "namespace": "statsbomb.event",
            "native_id": native_event_id,
        },
    )
    event_time = {
        "receive_at": fetched_at,
        "receive_basis": "upstream_exporter",
        "source_at": None,
        "publish_at": None,
        "exchange_at": None,
    }
    raw_ordinal = sequence - 1
    raw_payload = {
        "dataset_id": _DATASET_ID,
        "partition": f"events-{match_id}",
        "raw_object_hash": object_sha256,
        "raw_record_ordinal": raw_ordinal,
    }
    raw_material = {
        "envelope_version": "v0",
        "event_type": "raw_observation",
        "payload_schema_version": "v0",
        "source": {
            "system": _SOURCE,
            "stream": "events",
            "venue": None,
            "sequence": raw_ordinal,
            "capture_session_id": f"static:{object_sha256}",
            "record_ordinal": raw_ordinal,
        },
        "time": event_time,
        "canonical_refs": canonical_refs,
        "native_refs": native_refs,
        "lineage": {
            "raw_object_hash": object_sha256,
            "raw_record_ordinal": raw_ordinal,
            "parent_event_ids": (),
        },
        "experiment_id": None,
        "rule_snapshot_ref": None,
        "quality_flags": quality_flags,
        "payload": raw_payload,
        "payload_sha256": payload_sha256(raw_payload),
    }
    raw_event_id = event_id_for(raw_material)
    normalized_material = {
        "envelope_version": "v0",
        "event_type": "normalized_observation",
        "payload_schema_version": "v0",
        "source": {
            "system": _SOURCE,
            "stream": "events.normalized",
            "venue": None,
            "sequence": sequence,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        "time": event_time,
        "canonical_refs": canonical_refs,
        "native_refs": native_refs,
        "lineage": {
            "raw_object_hash": None,
            "raw_record_ordinal": None,
            "parent_event_ids": (raw_event_id,),
        },
        "experiment_id": _EXPERIMENT_ID,
        "rule_snapshot_ref": None,
        "quality_flags": quality_flags,
        "payload": payload,
        "payload_sha256": payload_sha256(payload),
    }
    return event_id_for(normalized_material)


def _domain_event(
    *,
    payload: dict[str, object],
    event_id: str,
) -> SoccerGameEvent:
    values = dict(payload)
    if values.pop("sport") != "soccer":
        raise SoccerSeasonCensusError("normalized event sport must be soccer")
    lineup = values.get("lineup_player_ids")
    flags = values.get("quality_flags")
    if type(lineup) is not list or type(flags) is not list:
        raise SoccerSeasonCensusError(
            "normalized event collections are not canonical lists"
        )
    values["lineup_player_ids"] = tuple(lineup)
    values["quality_flags"] = tuple(flags)
    return SoccerGameEvent(event_id=event_id, **values)  # type: ignore[arg-type]


def _census_once(
    *,
    matches: Sequence[Mapping[str, Any]],
    load_events: EventLoader,
) -> SoccerSeasonCensusRun:
    completed_games = 0
    source_events = 0
    events_reduced = 0
    clock_regressions = 0
    flag_counts: Counter[str] = Counter()
    score_mismatch_match_ids: list[int] = []
    failures: list[SoccerSeasonCensusFailure] = []
    state_hashes: list[dict[str, object]] = []

    for match in matches:
        match_id = _required_match_int(match, "match_id", minimum=1)
        home_team_id = _team_id(match, "home")
        away_team_id = _team_id(match, "away")
        expected_home_score = _required_match_int(match, "home_score")
        expected_away_score = _required_match_int(match, "away_score")
        partition = load_events(match_id)
        if partition.match_id != match_id:
            raise SoccerSeasonCensusError(
                "event partition match_id does not match requested match"
            )
        source_events += len(partition.events)
        game_id = f"game_statsbomb_{match_id}"
        state = initial_soccer_game_state(
            game_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )
        failed_sequence: int | None = None
        try:
            for raw_event in partition.events:
                raw_sequence = raw_event.get("index")
                failed_sequence = (
                    raw_sequence if type(raw_sequence) is int else None
                )
                flags: set[str] = set()
                if _has_source_coordinate_out_of_bounds(raw_event):
                    flags.add(_COORDINATE_FLAG)
                quality_flags = tuple(sorted(flags))
                payload = statsbomb_event_payload(
                    raw_event,
                    game_id=game_id,
                    quality_flags=quality_flags,
                )
                period = payload["period"]
                clock_ms = payload["clock_ms"]
                period_clock_ms = payload["period_clock_ms"]
                if (
                    period == state.period
                    and (
                        clock_ms < state.clock_ms  # type: ignore[operator]
                        or period_clock_ms < state.period_clock_ms  # type: ignore[operator]
                    )
                ):
                    flags.update(("clock_jump", "out_of_order"))
                    clock_regressions += 1
                    quality_flags = tuple(sorted(flags))
                    payload = statsbomb_event_payload(
                        raw_event,
                        game_id=game_id,
                        quality_flags=quality_flags,
                    )
                flag_counts.update(quality_flags)
                event_id = _event_envelope_id(
                    payload=payload,
                    object_sha256=partition.object_sha256,
                    fetched_at=partition.fetched_at,
                    match_id=match_id,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    quality_flags=quality_flags,
                )
                event = _domain_event(payload=payload, event_id=event_id)
                state = reduce_soccer_game_state(state, event)
                events_reduced += 1
        except (SoccerGameStateError, ValueError, TypeError) as error:
            failures.append(
                SoccerSeasonCensusFailure(
                    match_id=match_id,
                    failed_sequence=failed_sequence,
                    error=str(error),
                )
            )
            continue

        completed_games += 1
        if (
            state.home_score != expected_home_score
            or state.away_score != expected_away_score
        ):
            score_mismatch_match_ids.append(match_id)
        state_hashes.append(
            {
                "match_id": match_id,
                "final_state_sha256": state.state_sha256,
            }
        )

    state_hashes.sort(key=lambda item: int(item["match_id"]))
    return SoccerSeasonCensusRun(
        games_total=len(matches),
        completed_games=completed_games,
        fail_closed_games=len(failures),
        source_events=source_events,
        events_reduced=events_reduced,
        score_mismatches=len(score_mismatch_match_ids),
        score_mismatch_match_ids=tuple(sorted(score_mismatch_match_ids)),
        clock_regressions=clock_regressions,
        quality_flag_counts=tuple(sorted(flag_counts.items())),
        canonical_state_sha256=canonical_sha256(
            {"game_final_states": state_hashes}
        ),
        failures=tuple(failures),
    )


def census_loaded_statsbomb_season(
    *,
    matches: Sequence[Mapping[str, Any]],
    load_events: EventLoader,
    scan_runs: int = 2,
) -> SoccerSeasonCensusReport:
    """Replay loaded immutable partitions repeatedly and compare complete results."""

    if (
        isinstance(matches, (str, bytes))
        or not isinstance(matches, Sequence)
        or not matches
    ):
        raise SoccerSeasonCensusError("matches must be a nonempty sequence")
    if type(scan_runs) is not int or scan_runs < 2:
        raise SoccerSeasonCensusError("scan_runs must be an integer >= 2")
    summaries = tuple(
        _census_once(matches=matches, load_events=load_events)
        for _ in range(scan_runs)
    )
    deterministic = all(summary == summaries[0] for summary in summaries[1:])
    return SoccerSeasonCensusReport(
        census_version="v0",
        dataset_id=_DATASET_ID,
        source_version=STATSBOMB_COMMIT,
        scan_runs=scan_runs,
        deterministic=deterministic,
        canonical_state_sha256=summaries[0].canonical_state_sha256,
        run_summaries=summaries,
    )


def _one_manifest(directory: Path, *, context: str) -> Path:
    paths = tuple(sorted(directory.glob("*.manifest.json")))
    if len(paths) != 1:
        raise SoccerSeasonCensusError(
            f"{context} requires exactly one manifest; found {len(paths)}"
        )
    return paths[0]


def run_frozen_statsbomb_season_census(
    *,
    program_root: str | Path,
    scan_runs: int = 2,
) -> SoccerSeasonCensusReport:
    """Verify and replay all 380 frozen Premier League 2015/16 partitions."""

    root = Path(program_root).resolve()
    store_root = root / "var" / "raw"
    manifest_base = (
        store_root
        / "manifests"
        / f"source={_SOURCE}"
        / f"dataset={_DATASET_ID}"
        / f"version={STATSBOMB_COMMIT}"
    )
    index_path = _one_manifest(
        manifest_base / "partition=matches-2-27",
        context="StatsBomb match index",
    )
    verified_index = read_verified_static_object(
        index_path,
        store_root=store_root,
        program_root=root,
    )
    index_audit = inspect_statsbomb_match_index(verified_index.object_bytes)
    try:
        loaded_matches = json.loads(verified_index.object_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SoccerSeasonCensusError(
            "verified StatsBomb match index is not JSON"
        ) from error
    if type(loaded_matches) is not list:
        raise SoccerSeasonCensusError("verified StatsBomb match index is not an array")
    matches = tuple(loaded_matches)
    if len(matches) != STATSBOMB_EXPECTED_MATCHES:
        raise SoccerSeasonCensusError(
            f"frozen census requires {STATSBOMB_EXPECTED_MATCHES} matches"
        )

    manifest_by_match: dict[int, Path] = {}
    for match_id in index_audit.match_ids:
        manifest_by_match[match_id] = _one_manifest(
            manifest_base / f"partition=events-{match_id}",
            context=f"StatsBomb events-{match_id}",
        )

    def load_events(match_id: int) -> FrozenStatsBombEventPartition:
        verified = read_verified_static_object(
            manifest_by_match[match_id],
            store_root=store_root,
            program_root=root,
        )
        audit = inspect_statsbomb_event(
            verified.object_bytes,
            match_id=match_id,
        )
        try:
            loaded_events = json.loads(verified.object_bytes)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise SoccerSeasonCensusError(
                f"verified StatsBomb events-{match_id} is not JSON"
            ) from error
        if type(loaded_events) is not list or len(loaded_events) != audit.event_count:
            raise SoccerSeasonCensusError(
                f"verified StatsBomb events-{match_id} changed after inspection"
            )
        if any(type(event) is not dict for event in loaded_events):
            raise SoccerSeasonCensusError(
                f"verified StatsBomb events-{match_id} contains a non-object"
            )
        return FrozenStatsBombEventPartition(
            match_id=match_id,
            object_sha256=verified.record.manifest.object_sha256,
            fetched_at=verified.record.manifest.fetched_at,
            events=tuple(loaded_events),
        )

    return census_loaded_statsbomb_season(
        matches=matches,
        load_events=load_events,
        scan_runs=scan_runs,
    )


def _report_passes(report: SoccerSeasonCensusReport) -> bool:
    return report.deterministic and all(
        summary.completed_games == summary.games_total
        and summary.fail_closed_games == 0
        and not summary.failures
        and summary.events_reduced == summary.source_events
        and summary.score_mismatches == 0
        and not summary.score_mismatch_match_ids
        for summary in report.run_summaries
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the frozen StatsBomb PL 2015/16 season twice."
    )
    parser.add_argument(
        "--program-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--scan-runs", type=int, default=2)
    arguments = parser.parse_args()
    report = run_frozen_statsbomb_season_census(
        program_root=arguments.program_root,
        scan_runs=arguments.scan_runs,
    )
    print(
        json.dumps(
            report.to_document(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if _report_passes(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FrozenStatsBombEventPartition",
    "SoccerSeasonCensusError",
    "SoccerSeasonCensusFailure",
    "SoccerSeasonCensusReport",
    "SoccerSeasonCensusRun",
    "census_loaded_statsbomb_season",
    "run_frozen_statsbomb_season_census",
]
