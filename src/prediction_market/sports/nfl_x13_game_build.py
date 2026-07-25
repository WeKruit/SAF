"""Deterministic twenty-game X-13 replay, audit, and derived publication."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from prediction_market.sports.nfl_season_census import (
    census_loaded_nflverse_season,
)
from prediction_market.sports.nfl_x13_dataset import (
    X13SourceBundle,
    load_x13_source_bundle,
)
from prediction_market.sports.nfl_x13_replay import (
    CanonicalGameEventV1,
    CumulativeStatLedgerEntryV1,
    FinalizedEpisodeV1,
    X13GameLedgerV1,
    build_x13_game_ledger,
)
from prediction_market.sports.nfl_x13_spec import (
    X13_BATCH_SPEC,
    X13_GAME_BINDINGS,
    X13_GAME_IDS,
    canonical_sha256,
)
from prediction_market.sports.x11 import X11_NFLVERSE_VERSION


_SHA256_PREFIX = "sha256:"
_BUILDER_VERSION = "nfl-x13-game-state-v1"
_RUNTIME_SEMANTIC_FIELDS = frozenset(
    {
        "compute_workers",
        "census_transition_count",
    }
)
_RUNTIME_TELEMETRY_FIELDS = frozenset(
    {
        "elapsed_ns",
        "peak_rss_bytes",
        "census_latency_p50_ns",
        "census_latency_p95_ns",
        "census_latency_p99_ns",
        "ledger_runtime_total_ns",
    }
)
_RUNTIME_FIELDS = _RUNTIME_SEMANTIC_FIELDS | _RUNTIME_TELEMETRY_FIELDS


class X13GameBuildError(ValueError):
    """The twenty-game replay cannot prove its frozen audit gates."""


@dataclass(frozen=True, slots=True)
class X13GameStateArtifact:
    game_id: str
    events: tuple[CanonicalGameEventV1, ...]
    episodes: tuple[FinalizedEpisodeV1, ...]
    stat_ledger: tuple[CumulativeStatLedgerEntryV1, ...]
    events_sha256: str
    ledger_sha256: str
    source_time_present_rows: int
    participation_status_counts: Mapping[str, int]
    final_score: Mapping[str, int]
    final_score_verified: bool
    personnel_summary: Mapping[str, object]
    runtime_ns: int


@dataclass(frozen=True, slots=True)
class X13GameStateBuild:
    games: Mapping[str, X13GameStateArtifact]
    census: Mapping[str, object]
    source: X13SourceBundle
    manifest: Mapping[str, object]


def _require_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value.startswith(_SHA256_PREFIX)
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise X13GameBuildError(f"{field} must be a lowercase SHA-256")
    return value


def _build_identity_material(
    manifest_without_build_id: Mapping[str, object],
) -> dict[str, object]:
    """Keep performance telemetry outside the deterministic data identity."""

    material = dict(manifest_without_build_id)
    runtime = material.get("runtime")
    if isinstance(runtime, Mapping):
        material["runtime"] = {
            key: value
            for key, value in runtime.items()
            if key not in _RUNTIME_TELEMETRY_FIELDS
        }
    return material


def _validate_runtime(runtime: object) -> dict[str, int]:
    if not isinstance(runtime, Mapping) or set(runtime) != _RUNTIME_FIELDS:
        raise X13GameBuildError("runtime must contain the exact v1 fields")
    validated: dict[str, int] = {}
    for field in sorted(_RUNTIME_FIELDS):
        value = runtime.get(field)
        if type(value) is not int or value <= 0:
            raise X13GameBuildError(f"runtime.{field} must be positive")
        validated[field] = value
    if validated["compute_workers"] != 2:
        raise X13GameBuildError("X-13 game build requires exactly two workers")
    return validated


def _validate_runtime_against_census(
    runtime: object,
    census: object,
) -> dict[str, int]:
    validated = _validate_runtime(runtime)
    if (
        not isinstance(census, Mapping)
        or type(census.get("transitions")) is not int
        or validated["census_transition_count"] != census["transitions"]
    ):
        raise X13GameBuildError("hash graph census transition count changed")
    return validated


def x13_game_state_manifest_semantic_sha256(
    manifest: Mapping[str, object],
) -> str:
    """Validate and return the replay identity, excluding measured runtime."""

    if not isinstance(manifest, Mapping):
        raise X13GameBuildError("game-state manifest must be a mapping")
    document = dict(manifest)
    declared = document.pop("build_id", None)
    _validate_runtime(document.get("runtime"))
    computed = canonical_sha256(_build_identity_material(document))
    if declared != computed:
        raise X13GameBuildError(
            "game-state manifest does not match its semantic identity"
        )
    return computed


def _census_identity_material(census: Mapping[str, object]) -> dict[str, object]:
    """Hash audit facts independently from measured reducer performance."""

    return {
        key: value
        for key, value in census.items()
        if key != "reducer_latency"
    }


def build_x13_game_state_manifest(
    *,
    ledger_hashes: Mapping[str, str],
    source_manifest_hashes: tuple[str, ...],
    census_sha256: str,
    builder_version: str,
    runtime: Mapping[str, int],
) -> dict[str, object]:
    """Calculate the immutable game-state build identity."""

    if (
        not isinstance(ledger_hashes, Mapping)
        or set(ledger_hashes) != set(X13_GAME_IDS)
        or len(ledger_hashes) != 20
    ):
        raise X13GameBuildError("ledger hashes must cover the exact frozen 20")
    if type(source_manifest_hashes) is not tuple or not source_manifest_hashes:
        raise X13GameBuildError("source_manifest_hashes must be a nonempty tuple")
    for game_id, digest in ledger_hashes.items():
        _require_sha256(digest, f"{game_id}.ledger_sha256")
    for index, digest in enumerate(source_manifest_hashes):
        _require_sha256(digest, f"source_manifest_hashes[{index}]")
    _require_sha256(census_sha256, "census_sha256")
    if type(builder_version) is not str or not builder_version:
        raise X13GameBuildError("builder_version must be nonempty")
    validated_runtime = _validate_runtime(runtime)

    ordered_hashes = {
        game_id: ledger_hashes[game_id] for game_id in X13_GAME_IDS
    }
    manifest = {
        "schema": "nfl_x13_game_state_build_manifest_v1",
        "experiment_id": "X-13",
        "status": "PRELIMINARY_SOURCE_TIME_ONLY",
        "plan_id": X13_BATCH_SPEC.plan_id,
        "builder_version": builder_version,
        "game_count": 20,
        "game_ids": list(X13_GAME_IDS),
        "ledger_hashes": ordered_hashes,
        "source_manifest_hashes": sorted(source_manifest_hashes),
        "census_sha256": census_sha256,
        "deterministic_two_run_hashes_equal": True,
        "publication_gate": "PASS",
        "runtime": validated_runtime,
        "evidence_boundary": {
            "game_source_replay": True,
            "personnel_research_only": True,
            "tracking_coordinate": False,
            "exact_substitution_time": False,
            "market_alignment": False,
            "causal_or_alpha_claim": False,
        },
    }
    return {
        **manifest,
        "build_id": canonical_sha256(_build_identity_material(manifest)),
    }


def _build_ledgers(
    source: X13SourceBundle,
) -> tuple[dict[str, X13GameLedgerV1], dict[str, int]]:
    timings: dict[str, int] = {}

    def build_one(game_id: str) -> tuple[str, X13GameLedgerV1, int]:
        started = time.perf_counter_ns()
        ledger = build_x13_game_ledger(
            source.pbp_rows_by_game[game_id],
            source.participation_rows_by_game[game_id],
        )
        return game_id, ledger, time.perf_counter_ns() - started

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="x13-replay") as pool:
        results = tuple(pool.map(build_one, X13_GAME_IDS))
    ledgers: dict[str, X13GameLedgerV1] = {}
    for game_id, ledger, elapsed in results:
        ledgers[game_id] = ledger
        timings[game_id] = elapsed
    return ledgers, timings


def _scores(events: tuple[CanonicalGameEventV1, ...]) -> dict[str, int]:
    if not events:
        raise X13GameBuildError("game has no canonical events")
    game_id = events[0].game_id
    home_team = events[0].home_team
    away_team = events[0].away_team
    if home_team is None or away_team is None:
        raise X13GameBuildError("game event stream lacks team identity")
    if any(
        event.game_id != game_id
        or event.home_team != home_team
        or event.away_team != away_team
        for event in events
    ):
        raise X13GameBuildError("game event stream changed game or team identity")
    terminal_events = [event for event in events if event.game_end]
    if len(terminal_events) != 1 or terminal_events[0] is not events[-1]:
        raise X13GameBuildError(
            "game event stream must end with exactly one game_end event"
        )
    terminal_event = terminal_events[0]
    terminal_state = next(
        (
            state
            for state in (terminal_event.post_state, terminal_event.pre_state)
            if state is not None
            and state.terminal
            and state.home_score is not None
            and state.away_score is not None
        ),
        None,
    )
    if terminal_state is None:
        raise X13GameBuildError("game_end event lacks a terminal scoreboard")
    return {
        away_team: terminal_state.away_score,
        home_team: terminal_state.home_score,
    }


def _game_artifact(
    game_id: str,
    ledger: X13GameLedgerV1,
    source: X13SourceBundle,
    runtime_ns: int,
) -> X13GameStateArtifact:
    binding = next(
        binding
        for binding in X13_GAME_BINDINGS
        if binding.native_game_id == game_id
    )
    if any(event.game_id != game_id for event in ledger.events):
        raise X13GameBuildError(f"{game_id} ledger contains a foreign event")
    if len(ledger.events) != len(source.pbp_rows_by_game[game_id]):
        raise X13GameBuildError(f"{game_id} replay dropped or added source rows")
    observed_scores = _scores(ledger.events)
    expected_scores = {
        binding.away_team: binding.away_score,
        binding.home_team: binding.home_score,
    }
    if observed_scores != expected_scores:
        raise X13GameBuildError(
            f"{game_id} final score changed: {observed_scores!r}"
        )
    statuses = Counter(event.participation_status for event in ledger.events)
    personnel_rows = len(source.participation_rows_by_game[game_id])
    full_rows = statuses.get("present_complete", 0)
    partial_rows = statuses.get("present_partial", 0)
    missing_rows = statuses.get("missing", 0)
    if full_rows + partial_rows + missing_rows != len(ledger.events):
        raise X13GameBuildError(f"{game_id} personnel coverage is inconsistent")
    return X13GameStateArtifact(
        game_id=game_id,
        events=ledger.events,
        episodes=ledger.episodes,
        stat_ledger=ledger.stat_ledger,
        events_sha256=ledger.events_sha256,
        ledger_sha256=ledger.artifact_sha256,
        source_time_present_rows=sum(
            event.source_time_status != "missing" for event in ledger.events
        ),
        participation_status_counts=MappingProxyType(dict(sorted(statuses.items()))),
        final_score=MappingProxyType(observed_scores),
        final_score_verified=True,
        personnel_summary=MappingProxyType(
            {
                "license_status": "research_only",
                "semantics": (
                    "observed on-field roster per play; adjacent set differences "
                    "are between_play_roster_change, not exact substitution time"
                ),
                "source_participation_rows": personnel_rows,
                "present_complete_11v11_rows": full_rows,
                "present_partial_rows": partial_rows,
                "missing_rows": missing_rows,
            }
        ),
        runtime_ns=runtime_ns,
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def build_x13_game_state(
    *,
    program_root: str | Path,
    raw_store_root: str | Path,
) -> X13GameStateBuild:
    """Verify, audit, and replay all twenty games twice with two workers."""

    started = time.perf_counter_ns()
    source = load_x13_source_bundle(
        program_root=program_root,
        raw_store_root=raw_store_root,
    )
    selected_rows = [
        row for game_id in X13_GAME_IDS for row in source.pbp_rows_by_game[game_id]
    ]
    census_report = census_loaded_nflverse_season(
        rows=selected_rows,
        raw_object_sha256=source.pbp_object_sha256,
        source_version=X11_NFLVERSE_VERSION,
        scan_runs=2,
    )
    if (
        census_report.games_total != 20
        or census_report.completed_games != 20
        or census_report.fail_closed_games != 0
        or not census_report.deterministic
        or census_report.failures
        or any(audit.mismatches for audit in census_report.field_audits)
    ):
        raise X13GameBuildError(
            "20-game state census did not pass all deterministic invariants"
        )
    first, first_runtime = _build_ledgers(source)
    second, _ = _build_ledgers(source)
    first_hashes = {
        game_id: first[game_id].artifact_sha256 for game_id in X13_GAME_IDS
    }
    second_hashes = {
        game_id: second[game_id].artifact_sha256 for game_id in X13_GAME_IDS
    }
    if first_hashes != second_hashes:
        raise X13GameBuildError("two ledger runs produced different canonical hashes")

    games = {
        game_id: _game_artifact(
            game_id,
            first[game_id],
            source,
            first_runtime[game_id],
        )
        for game_id in X13_GAME_IDS
    }
    census = census_report.to_document()
    census_sha256 = canonical_sha256(_census_identity_material(census))
    runtime = {
        "elapsed_ns": max(1, time.perf_counter_ns() - started),
        "peak_rss_bytes": max(1, _peak_rss_bytes()),
        "compute_workers": 2,
        "census_transition_count": census_report.transitions,
        "census_latency_p50_ns": census_report.reducer_latency["p50_ns"],
        "census_latency_p95_ns": census_report.reducer_latency["p95_ns"],
        "census_latency_p99_ns": census_report.reducer_latency["p99_ns"],
        "ledger_runtime_total_ns": sum(first_runtime.values()),
    }
    manifest = build_x13_game_state_manifest(
        ledger_hashes=first_hashes,
        source_manifest_hashes=(
            source.pbp_manifest_sha256,
            source.participation_manifest_sha256,
        ),
        census_sha256=census_sha256,
        builder_version=_BUILDER_VERSION,
        runtime=runtime,
    )
    return X13GameStateBuild(
        games=MappingProxyType(games),
        census=MappingProxyType(census),
        source=source,
        manifest=MappingProxyType(manifest),
    )


def _jsonable_stat_ledger(
    entries: tuple[CumulativeStatLedgerEntryV1, ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in entries:
        value = asdict(entry)
        value["period_scores"] = {
            str(period): scores
            for period, scores in entry.period_scores.items()
        }
        result.append(value)
    return result


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise X13GameBuildError("derived artifact is not canonical JSON") from error


def _write_new(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(contents)
        offset = 0
        while offset < len(view):
            count = os.write(descriptor, view[offset:])
            if count <= 0:
                raise OSError("short write")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _game_manifest(game: X13GameStateArtifact, source: X13SourceBundle) -> dict[str, object]:
    return {
        "schema": "nfl_x13_game_state_artifact_manifest_v1",
        "experiment_id": "X-13",
        "status": "PRELIMINARY_SOURCE_TIME_ONLY",
        "game_id": game.game_id,
        "events_sha256": game.events_sha256,
        "ledger_sha256": game.ledger_sha256,
        "event_count": len(game.events),
        "episode_count": len(game.episodes),
        "stat_ledger_count": len(game.stat_ledger),
        "final_score": dict(game.final_score),
        "final_score_verified": game.final_score_verified,
        "source_time_present_rows": game.source_time_present_rows,
        "participation_status_counts": dict(game.participation_status_counts),
        "source_manifest_hashes": sorted(
            [
                source.pbp_manifest_sha256,
                source.participation_manifest_sha256,
            ]
        ),
        "builder_version": _BUILDER_VERSION,
    }


def publish_x13_game_state(
    build: X13GameStateBuild,
    *,
    output_root: str | Path,
) -> Path:
    """Publish all 20 derived game-state sets atomically."""

    if not isinstance(build, X13GameStateBuild):
        raise X13GameBuildError("build must be X13GameStateBuild")
    if set(build.games) != set(X13_GAME_IDS):
        raise X13GameBuildError("publication requires the exact frozen 20")
    root = Path(output_root).resolve()
    if root == Path(root.anchor):
        raise X13GameBuildError("output_root cannot be a filesystem root")
    root.mkdir(parents=True, exist_ok=True)
    semantic_sha256 = x13_game_state_manifest_semantic_sha256(build.manifest)
    target = root / semantic_sha256.replace(":", "-")
    if target.exists():
        verified = verify_x13_game_state_publication(target)
        existing = json.loads((target / "build_manifest.json").read_text("utf-8"))
        if (
            verified.get("build_id") != semantic_sha256
            or x13_game_state_manifest_semantic_sha256(existing)
            != semantic_sha256
        ):
            raise X13GameBuildError("immutable build target conflicts")
        return target

    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=root)
    )
    try:
        games_root = stage / "games"
        games_root.mkdir()
        for game_id in X13_GAME_IDS:
            game = build.games[game_id]
            directory = games_root / game_id
            directory.mkdir()
            _write_new(
                directory / "canonical_events.json",
                _canonical_bytes([asdict(event) for event in game.events]) + b"\n",
            )
            _write_new(
                directory / "finalized_episodes.json",
                _canonical_bytes([asdict(episode) for episode in game.episodes])
                + b"\n",
            )
            _write_new(
                directory / "stat_ledger.json",
                _canonical_bytes(_jsonable_stat_ledger(game.stat_ledger)) + b"\n",
            )
            _write_new(
                directory / "personnel_summary.json",
                _canonical_bytes(dict(game.personnel_summary)) + b"\n",
            )
            _write_new(
                directory / "game_manifest.json",
                _canonical_bytes(_game_manifest(game, build.source)) + b"\n",
            )
            _fsync(directory)
        index = {
            game_id: {
                "path": f"games/{game_id}",
                "events_sha256": build.games[game_id].events_sha256,
                "ledger_sha256": build.games[game_id].ledger_sha256,
            }
            for game_id in X13_GAME_IDS
        }
        _write_new(stage / "index.json", _canonical_bytes(index) + b"\n")
        _write_new(
            stage / "census.json",
            _canonical_bytes(dict(build.census)) + b"\n",
        )
        _write_new(
            stage / "build_manifest.json",
            _canonical_bytes(dict(build.manifest)) + b"\n",
        )
        files = sorted(
            path
            for path in stage.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        )
        checksums = "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(stage).as_posix()}"
            for path in files
        )
        _write_new(stage / "checksums.sha256", (checksums + "\n").encode("ascii"))
        _fsync(games_root)
        _fsync(stage)
        os.replace(stage, target)
        _fsync(root)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    verify_x13_game_state_publication(target)
    return target


def verify_x13_game_state_publication(path: str | Path) -> dict[str, object]:
    """Verify every byte and recompute the complete publication hash graph."""

    root = Path(path).resolve()
    ledger = root / "checksums.sha256"
    if not root.is_dir() or root.is_symlink() or not ledger.is_file():
        raise X13GameBuildError("game-state publication is incomplete")
    expected: dict[str, str] = {}
    for line in ledger.read_text("ascii").splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise X13GameBuildError("checksum ledger is malformed")
        digest, relative_text = line[:64], line[66:]
        relative = PurePosixPath(relative_text)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in expected
        ):
            raise X13GameBuildError("checksum ledger entry is invalid")
        expected[relative_text] = digest
    actual = {
        child.relative_to(root).as_posix()
        for child in root.rglob("*")
        if child.is_file() and child.name != "checksums.sha256"
    }
    if set(expected) != actual:
        raise X13GameBuildError("checksum file set does not match publication")
    for relative, digest in expected.items():
        child = root.joinpath(*PurePosixPath(relative).parts)
        if child.is_symlink() or hashlib.sha256(child.read_bytes()).hexdigest() != digest:
            raise X13GameBuildError(f"checksum mismatch: {relative}")

    def read_canonical(relative: str) -> Any:
        child = root.joinpath(*PurePosixPath(relative).parts)
        try:
            raw = child.read_bytes()
            value = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise X13GameBuildError(
                f"hash graph contains unreadable JSON: {relative}"
            ) from error
        if raw != _canonical_bytes(value) + b"\n":
            raise X13GameBuildError(
                f"hash graph contains non-canonical JSON: {relative}"
            )
        return value

    manifest = read_canonical("build_manifest.json")
    if not isinstance(manifest, dict):
        raise X13GameBuildError("hash graph build manifest must be an object")
    try:
        declared_build_id = x13_game_state_manifest_semantic_sha256(manifest)
    except X13GameBuildError as error:
        raise X13GameBuildError(
            "hash graph build manifest does not prove the frozen build"
        ) from error
    if (
        manifest.get("game_ids") != list(X13_GAME_IDS)
        or manifest.get("game_count") != 20
        or manifest.get("publication_gate") != "PASS"
        or manifest.get("experiment_id") != "X-13"
        or manifest.get("status") != "PRELIMINARY_SOURCE_TIME_ONLY"
        or manifest.get("plan_id") != X13_BATCH_SPEC.plan_id
        or manifest.get("deterministic_two_run_hashes_equal") is not True
        or root.name != str(declared_build_id).replace(":", "-")
    ):
        raise X13GameBuildError(
            "hash graph build manifest does not prove the frozen build"
        )
    source_manifest_hashes = manifest.get("source_manifest_hashes")
    if (
        not isinstance(source_manifest_hashes, list)
        or not source_manifest_hashes
        or any(
            _require_sha256(value, "source manifest") != value
            for value in source_manifest_hashes
        )
    ):
        raise X13GameBuildError("hash graph source manifests are invalid")
    census = read_canonical("census.json")
    if (
        canonical_sha256(_census_identity_material(census))
        != manifest.get("census_sha256")
    ):
        raise X13GameBuildError("hash graph census hash changed")
    _validate_runtime_against_census(manifest.get("runtime"), census)
    index = read_canonical("index.json")
    if not isinstance(index, dict):
        raise X13GameBuildError("hash graph index must be an object")

    bindings = {
        binding.native_game_id: binding for binding in X13_GAME_BINDINGS
    }
    recomputed_ledger_hashes: dict[str, str] = {}
    recomputed_index: dict[str, dict[str, str]] = {}
    for game_id in X13_GAME_IDS:
        directory = root / "games" / game_id
        actual_game_files = {
            child.name for child in directory.iterdir() if child.is_file()
        }
        required_game_files = {
            "canonical_events.json",
            "finalized_episodes.json",
            "stat_ledger.json",
            "personnel_summary.json",
            "game_manifest.json",
        }
        if actual_game_files != required_game_files:
            raise X13GameBuildError(
                f"hash graph {game_id} does not have the exact derived artifacts"
            )
        prefix = f"games/{game_id}"
        events = read_canonical(f"{prefix}/canonical_events.json")
        episodes = read_canonical(f"{prefix}/finalized_episodes.json")
        stat_ledger = read_canonical(f"{prefix}/stat_ledger.json")
        personnel = read_canonical(f"{prefix}/personnel_summary.json")
        game_manifest = read_canonical(f"{prefix}/game_manifest.json")
        if (
            not isinstance(events, list)
            or not events
            or not isinstance(episodes, list)
            or not isinstance(stat_ledger, list)
            or not isinstance(personnel, dict)
            or not isinstance(game_manifest, dict)
        ):
            raise X13GameBuildError(
                f"hash graph {game_id} artifact shapes are invalid"
            )

        binding = bindings[game_id]
        if any(
            not isinstance(event, dict)
            or event.get("game_id") != game_id
            or event.get("away_team") != binding.away_team
            or event.get("home_team") != binding.home_team
            for event in events
        ):
            raise X13GameBuildError(
                f"hash graph {game_id} event identity changed"
            )
        if any(
            not isinstance(episode, dict)
            or episode.get("game_id") != game_id
            for episode in episodes
        ) or any(
            not isinstance(entry, dict) or entry.get("game_id") != game_id
            for entry in stat_ledger
        ):
            raise X13GameBuildError(
                f"hash graph {game_id} ledger identity changed"
            )

        final_event = events[-1]
        terminal_state = next(
            (
                state
                for state in (
                    final_event.get("post_state"),
                    final_event.get("pre_state"),
                )
                if isinstance(state, dict)
                and state.get("terminal") is True
                and type(state.get("home_score")) is int
                and type(state.get("away_score")) is int
            ),
            None,
        )
        if final_event.get("game_end") is not True or terminal_state is None:
            raise X13GameBuildError(
                f"hash graph {game_id} lacks a terminal scoreboard"
            )
        final_score = {
            binding.away_team: terminal_state["away_score"],
            binding.home_team: terminal_state["home_score"],
        }
        if final_score != {
            binding.away_team: binding.away_score,
            binding.home_team: binding.home_score,
        }:
            raise X13GameBuildError(
                f"hash graph {game_id} final score changed"
            )

        events_sha256 = canonical_sha256(
            {
                "schema_version": "CanonicalGameEventV1",
                "events": events,
            }
        )
        ledger_sha256 = canonical_sha256(
            {
                "schema_version": "X13GameLedgerV1",
                "events": events,
                "episodes": episodes,
                "stat_ledger": stat_ledger,
            }
        )
        status_counts = dict(
            sorted(
                Counter(
                    str(event.get("participation_status"))
                    for event in events
                ).items()
            )
        )
        present_rows = (
            status_counts.get("present_complete", 0)
            + status_counts.get("present_partial", 0)
        )
        expected_personnel = {
            "license_status": "research_only",
            "semantics": (
                "observed on-field roster per play; adjacent set differences "
                "are between_play_roster_change, not exact substitution time"
            ),
            "source_participation_rows": present_rows,
            "present_complete_11v11_rows": status_counts.get(
                "present_complete", 0
            ),
            "present_partial_rows": status_counts.get("present_partial", 0),
            "missing_rows": status_counts.get("missing", 0),
        }
        expected_game_manifest = {
            "schema": "nfl_x13_game_state_artifact_manifest_v1",
            "experiment_id": "X-13",
            "status": "PRELIMINARY_SOURCE_TIME_ONLY",
            "game_id": game_id,
            "events_sha256": events_sha256,
            "ledger_sha256": ledger_sha256,
            "event_count": len(events),
            "episode_count": len(episodes),
            "stat_ledger_count": len(stat_ledger),
            "final_score": final_score,
            "final_score_verified": True,
            "source_time_present_rows": sum(
                event.get("source_time_status") != "missing"
                for event in events
            ),
            "participation_status_counts": status_counts,
            "source_manifest_hashes": source_manifest_hashes,
            "builder_version": manifest.get("builder_version"),
        }
        if personnel != expected_personnel or game_manifest != expected_game_manifest:
            raise X13GameBuildError(
                f"hash graph {game_id} derived summary changed"
            )
        recomputed_ledger_hashes[game_id] = ledger_sha256
        recomputed_index[game_id] = {
            "path": f"games/{game_id}",
            "events_sha256": events_sha256,
            "ledger_sha256": ledger_sha256,
        }

    if (
        recomputed_ledger_hashes != manifest.get("ledger_hashes")
        or recomputed_index != index
    ):
        raise X13GameBuildError("hash graph index or ledger hashes changed")
    return {
        "build_id": declared_build_id,
        "game_count": 20,
        "verified_file_count": len(expected) + 1,
    }


__all__ = [
    "X13GameBuildError",
    "X13GameStateArtifact",
    "X13GameStateBuild",
    "build_x13_game_state",
    "build_x13_game_state_manifest",
    "publish_x13_game_state",
    "verify_x13_game_state_publication",
    "x13_game_state_manifest_semantic_sha256",
]
