"""Reaction-closed NFL 2025 sports cleaning for the X-13 expansion set.

This module deliberately has no market-data dependency.  It reads the frozen
candidate game identities, verifies the frozen nflverse objects, reuses the
existing X-13 replay/row-disposition/time builders, and publishes immutable
single-game sports bundles before publishing an all-or-nothing batch index.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.sports.nfl_factor_lab_cleaning import (
    build_sports_row_disposition_ledger,
)
from prediction_market.sports.nfl_x13_association import (
    build_game_time_intervals_v1,
)
from prediction_market.sports.nfl_x13_replay import build_x13_game_ledger

REACTION_ACCESS = "CLOSED"
SCHEMA_VERSION = "nfl_historical_sports_clean_v1"
EXPECTED_CANDIDATE_COUNT = 234
EXPECTED_COVERAGE_OBJECT_SHA256 = (
    "sha256:432277924ccc8249333b48b96060662cf630bf1a72616e79d9e186bdb2de1588"
)
EXPECTED_PBP_MANIFEST_SHA256 = (
    "sha256:e06985cfae03b5967d8970be8a9f7a82699d72b549595c044d1056b553c1d99d"
)
EXPECTED_PBP_OBJECT_SHA256 = (
    "sha256:3730c4db2ab99d2dfc4017de975b7610c46c35301b9280b65c03de1b1c74265a"
)
EXPECTED_PARTICIPATION_MANIFEST_SHA256 = (
    "sha256:c5b2f2d56b3e04ce8ce6674e09a2571eef39bfefaeb16d2c06da50a332b1e291"
)
EXPECTED_PARTICIPATION_OBJECT_SHA256 = (
    "sha256:3b26b9487267f011fdbafbce133800ce964c963fdb8b27725abf65663cc46e0f"
)
DEFAULT_COVERAGE_MANIFEST = Path(
    "artifacts/market-observation/nfl/x13/candidate-coverage/"
    "temporal-all-dual-v1/manifests/"
    "6549c5003b6419fc422dbd27fb6a2663ae22f8bc9f0584274c3321a47666e2ac"
    ".manifest.json"
)
DEFAULT_PBP_MANIFEST = Path(
    "var/raw/manifests/source=nflverse/dataset=DS-NFLVERSE/"
    "version=github-release-58152862-20260212T102526Z/"
    "partition=season-2025/"
    "e06985cfae03b5967d8970be8a9f7a82699d72b549595c044d1056b553c1d99d"
    ".manifest.json"
)
DEFAULT_PARTICIPATION_MANIFEST = Path(
    "var/raw/manifests/source=nflverse/"
    "dataset=DS-NFLVERSE-PARTICIPATION/"
    "version=release-asset-2025-observed-20260724/"
    "partition=season-2025/"
    "c5b2f2d56b3e04ce8ce6674e09a2571eef39bfefaeb16d2c06da50a332b1e291"
    ".manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/market-observation/nfl/x13/"
    "sports-clean-expansion/reaction-closed-v1"
)


class HistoricalSportsCleanError(RuntimeError):
    """A frozen input, reconciliation, or publication gate failed."""


@dataclass(frozen=True, slots=True)
class VerifiedSource:
    dataset_id: str
    manifest_path: Path
    manifest_logical_sha256: str
    manifest_file_sha256: str
    object_path: Path
    object_sha256: str
    byte_length: int
    license_status: str


@dataclass(frozen=True, slots=True)
class CandidateCoverage:
    game_ids: tuple[str, ...]
    manifest_path: Path
    manifest_file_sha256: str
    object_path: Path
    object_sha256: str
    selection_basis: str


@dataclass(frozen=True, slots=True)
class PublishedGameBundle:
    game_id: str
    bundle_sha256: str
    manifest_path: Path
    manifest_sha256: str
    counts: Mapping[str, int]
    reused_checkpoint: bool
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PublishedSportsCleanBatch:
    scope: str
    game_count: int
    batch_sha256: str
    index_path: Path
    index_sha256: str
    games: tuple[PublishedGameBundle, ...]
    aggregate_counts: Mapping[str, int]
    elapsed_seconds: float


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    if isinstance(value, set):
        normalized = [_canonical_value(child) for child in value]
        return sorted(
            normalized,
            key=lambda child: json.dumps(
                child,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise HistoricalSportsCleanError(
        f"value is not canonical-JSON serializable: {type(value).__name__}"
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


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalSportsCleanError(f"cannot read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise HistoricalSportsCleanError(f"JSON object required: {path}")
    return payload


def _resolve_under(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise HistoricalSportsCleanError(
            f"path escapes declared root: {relative}"
        )
    return candidate


def _verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_length: int | None = None,
) -> None:
    if not path.is_file():
        raise HistoricalSportsCleanError(f"required file is absent: {path}")
    observed_length = path.stat().st_size
    if expected_length is not None and observed_length != expected_length:
        raise HistoricalSportsCleanError(
            f"byte length mismatch for {path}: "
            f"{observed_length} != {expected_length}"
        )
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise HistoricalSportsCleanError(
            f"SHA-256 mismatch for {path}: "
            f"{observed_sha256} != {expected_sha256}"
        )


def discover_main_checkout(project_root: Path) -> Path:
    """Return the main checkout that owns the shared ``.git`` directory."""

    project_root = project_root.resolve()
    git_entry = project_root / ".git"
    if git_entry.is_dir():
        return project_root
    if not git_entry.is_file():
        raise HistoricalSportsCleanError(
            f"cannot discover git checkout from {project_root}"
        )
    marker = git_entry.read_text(encoding="utf-8").strip()
    if not marker.startswith("gitdir: "):
        raise HistoricalSportsCleanError("worktree .git file is malformed")
    git_dir = Path(marker.removeprefix("gitdir: ")).expanduser().resolve()
    if git_dir.parent.name != "worktrees" or git_dir.parent.parent.name != ".git":
        raise HistoricalSportsCleanError(
            f"unexpected linked-worktree gitdir: {git_dir}"
        )
    return git_dir.parents[2]


def verify_candidate_coverage(
    project_root: Path,
    manifest_path: Path | None = None,
) -> CandidateCoverage:
    manifest_path = (
        _resolve_under(project_root, DEFAULT_COVERAGE_MANIFEST)
        if manifest_path is None
        else manifest_path.resolve()
    )
    manifest = _read_json(manifest_path)
    manifest_file_sha256 = _sha256_file(manifest_path)
    expected_name_sha = "sha256:" + manifest_path.name.removesuffix(
        ".manifest.json"
    )
    if manifest_file_sha256 != expected_name_sha:
        raise HistoricalSportsCleanError(
            "candidate coverage manifest filename does not match its bytes"
        )
    if manifest.get("candidate_game_count") != EXPECTED_CANDIDATE_COUNT:
        raise HistoricalSportsCleanError(
            "candidate coverage manifest is not the frozen 234-game scope"
        )
    if manifest.get("object_sha256") != EXPECTED_COVERAGE_OBJECT_SHA256:
        raise HistoricalSportsCleanError("candidate coverage object drifted")
    object_path = _resolve_under(project_root, str(manifest["object_path"]))
    _verify_file(
        object_path,
        expected_sha256=EXPECTED_COVERAGE_OBJECT_SHA256,
        expected_length=int(manifest["byte_length"]),
    )
    payload = _read_json(object_path)
    if (
        payload.get("schema") != "nfl_2025_candidate_temporal_coverage_v1"
        or payload.get("scope") != "all-dual"
        or payload.get("candidate_game_count") != EXPECTED_CANDIDATE_COUNT
    ):
        raise HistoricalSportsCleanError(
            "candidate coverage payload violated the frozen schema/scope"
        )
    selection_basis = str(payload.get("selection_basis", ""))
    if "no_score_price_size_volume_or_direction_retained" not in selection_basis:
        raise HistoricalSportsCleanError(
            "candidate selection is not demonstrably reaction-blind"
        )
    games = payload.get("games")
    if not isinstance(games, list):
        raise HistoricalSportsCleanError("candidate coverage games are absent")
    # Only identities are retained here.  Trade presence/count fields are not
    # propagated into this reaction-closed sports materialization.
    game_ids = tuple(
        sorted(
            str(row["game_id"])
            for row in games
            if isinstance(row, Mapping) and isinstance(row.get("game_id"), str)
        )
    )
    if (
        len(game_ids) != EXPECTED_CANDIDATE_COUNT
        or len(set(game_ids)) != EXPECTED_CANDIDATE_COUNT
    ):
        raise HistoricalSportsCleanError(
            "candidate game identities are incomplete or duplicated"
        )
    if any(not game_id.startswith("2025_") for game_id in game_ids):
        raise HistoricalSportsCleanError("candidate game is outside 2025 season")
    return CandidateCoverage(
        game_ids=game_ids,
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_file_sha256,
        object_path=object_path,
        object_sha256=EXPECTED_COVERAGE_OBJECT_SHA256,
        selection_basis=selection_basis,
    )


def _verify_source(
    data_root: Path,
    relative_manifest: Path,
    *,
    expected_dataset_id: str,
    expected_manifest_sha256: str,
    expected_object_sha256: str,
    accepted_license_statuses: set[str],
) -> VerifiedSource:
    manifest_path = _resolve_under(data_root, relative_manifest)
    manifest = _read_json(manifest_path)
    if manifest.get("manifest_sha256") != expected_manifest_sha256:
        raise HistoricalSportsCleanError(
            f"{expected_dataset_id} manifest identity drifted"
        )
    if manifest.get("dataset_id") != expected_dataset_id:
        raise HistoricalSportsCleanError(
            f"{expected_dataset_id} manifest has wrong dataset_id"
        )
    if manifest.get("object_sha256") != expected_object_sha256:
        raise HistoricalSportsCleanError(
            f"{expected_dataset_id} source object drifted"
        )
    license_status = str(manifest.get("license_status", ""))
    if license_status not in accepted_license_statuses:
        raise HistoricalSportsCleanError(
            f"{expected_dataset_id} license is not allowed: {license_status}"
        )
    native_path = manifest.get("native_object_path")
    if not isinstance(native_path, str):
        raise HistoricalSportsCleanError(
            f"{expected_dataset_id} native_object_path is absent"
        )
    object_path = _resolve_under(data_root / "var/raw", native_path)
    byte_length = int(manifest["byte_length"])
    _verify_file(
        object_path,
        expected_sha256=expected_object_sha256,
        expected_length=byte_length,
    )
    return VerifiedSource(
        dataset_id=expected_dataset_id,
        manifest_path=manifest_path,
        manifest_logical_sha256=expected_manifest_sha256,
        manifest_file_sha256=_sha256_file(manifest_path),
        object_path=object_path,
        object_sha256=expected_object_sha256,
        byte_length=byte_length,
        license_status=license_status,
    )


def verify_frozen_2025_sources(
    data_root: Path,
) -> tuple[VerifiedSource, VerifiedSource]:
    pbp = _verify_source(
        data_root,
        DEFAULT_PBP_MANIFEST,
        expected_dataset_id="DS-NFLVERSE",
        expected_manifest_sha256=EXPECTED_PBP_MANIFEST_SHA256,
        expected_object_sha256=EXPECTED_PBP_OBJECT_SHA256,
        accepted_license_statuses={"approved"},
    )
    participation = _verify_source(
        data_root,
        DEFAULT_PARTICIPATION_MANIFEST,
        expected_dataset_id="DS-NFLVERSE-PARTICIPATION",
        expected_manifest_sha256=EXPECTED_PARTICIPATION_MANIFEST_SHA256,
        expected_object_sha256=EXPECTED_PARTICIPATION_OBJECT_SHA256,
        accepted_license_statuses={"research_only"},
    )
    return pbp, participation


def _builder_code_sha256() -> str:
    functions = (
        build_x13_game_ledger,
        build_sports_row_disposition_ledger,
        build_game_time_intervals_v1,
    )
    material = []
    for function in functions:
        source = inspect.getsourcefile(function)
        if source is None:
            raise HistoricalSportsCleanError(
                f"cannot bind builder source for {function.__name__}"
            )
        path = Path(source)
        material.append(
            {
                "function": function.__name__,
                "module": function.__module__,
                "source_file_sha256": _sha256_file(path),
            }
        )
    return _canonical_sha256(material)


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [dict(row) for row in frame.to_dict(orient="records")]


def _load_sports_frames(
    pbp: VerifiedSource,
    participation: VerifiedSource,
    game_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    requested = list(game_ids)
    pbp_frame = pd.read_parquet(
        pbp.object_path,
        filters=[("game_id", "in", requested)],
    )
    participation_frame = pd.read_parquet(
        participation.object_path,
        filters=[("nflverse_game_id", "in", requested)],
    ).rename(columns={"nflverse_game_id": "game_id"})
    for field, frame in (
        ("PBP", pbp_frame),
        ("participation", participation_frame),
    ):
        if "game_id" not in frame:
            raise HistoricalSportsCleanError(f"{field} lacks game_id")
        frame["game_id"] = frame["game_id"].astype(str)
    observed = set(pbp_frame["game_id"])
    missing = sorted(set(requested) - observed)
    unexpected = sorted(observed - set(requested))
    if missing or unexpected:
        raise HistoricalSportsCleanError(
            f"PBP candidate coverage mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    if pbp_frame.duplicated(["game_id", "play_id"]).any():
        raise HistoricalSportsCleanError(
            "PBP source has duplicate (game_id, play_id)"
        )
    if participation_frame.duplicated(["game_id", "play_id"]).any():
        raise HistoricalSportsCleanError(
            "participation source has duplicate (game_id, play_id)"
        )
    return pbp_frame, participation_frame


def _parquet_scalar(value: object) -> object:
    normalized = _canonical_value(value)
    if isinstance(normalized, (dict, list)):
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return normalized


def _parquet_table(records: Sequence[Mapping[str, object]]) -> pa.Table:
    if not records:
        raise HistoricalSportsCleanError("cannot publish an empty stage")
    columns = tuple(records[0])
    if any(tuple(row) != columns for row in records):
        raise HistoricalSportsCleanError(
            "stage records do not share one ordered schema"
        )
    flattened = [
        {key: _parquet_scalar(value) for key, value in row.items()}
        for row in records
    ]
    return pa.Table.from_pylist(flattened)


def _publish_parquet_stage(
    output_root: Path,
    *,
    game_id: str,
    stage_name: str,
    schema_name: str,
    grain: str,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    table = _parquet_table(records)
    work = output_root / ".work" / "objects"
    work.mkdir(parents=True, exist_ok=True)
    temporary = work / f"{uuid.uuid4().hex}.parquet"
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=9,
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        object_sha256 = _sha256_file(temporary)
        digest = object_sha256.removeprefix("sha256:")
        relative = (
            Path("single-game")
            / game_id
            / "objects"
            / "sha256"
            / digest[:2]
            / f"{digest}.parquet"
        )
        target = _resolve_under(output_root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            _verify_file(
                target,
                expected_sha256=object_sha256,
                expected_length=temporary.stat().st_size,
            )
            temporary.unlink()
        else:
            os.replace(temporary, target)
        semantic_rows_sha256 = _canonical_sha256(
            {"schema": schema_name, "rows": list(records)}
        )
        schema_fingerprint = _sha256_bytes(
            table.schema.serialize().to_pybytes()
        )
        return {
            "stage": stage_name,
            "schema": schema_name,
            "grain": grain,
            "object_path": relative.as_posix(),
            "object_sha256": object_sha256,
            "byte_length": target.stat().st_size,
            "row_count": table.num_rows,
            "semantic_rows_sha256": semantic_rows_sha256,
            "schema_fingerprint": schema_fingerprint,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> str:
    encoded = _canonical_bytes(payload)
    expected_sha256 = _sha256_bytes(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_bytes(encoded)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    if path.exists():
        observed = path.read_bytes()
        if observed != encoded:
            temporary.unlink(missing_ok=True)
            raise HistoricalSportsCleanError(
                f"immutable JSON target already differs: {path}"
            )
        temporary.unlink()
    else:
        os.replace(temporary, path)
    _verify_file(
        path,
        expected_sha256=expected_sha256,
        expected_length=len(encoded),
    )
    return expected_sha256


def _stage_records_for_game(
    game_id: str,
    pbp_rows: Sequence[Mapping[str, object]],
    participation_rows: Sequence[Mapping[str, object]],
    source_hashes: Mapping[str, str],
) -> tuple[
    dict[str, list[dict[str, object]]],
    Mapping[str, str],
    int,
]:
    replay_rows, adjudications = _adjudicate_replay_rows(pbp_rows)
    ledger = build_x13_game_ledger(replay_rows, participation_rows)
    events = [asdict(row) for row in ledger.events]
    episodes = [asdict(row) for row in ledger.episodes]
    stats = [asdict(row) for row in ledger.stat_ledger]
    disposition_ledger = build_sports_row_disposition_ledger(
        selected_pbp_rows=pbp_rows,
        finalized_episode_rows=episodes,
        canonical_event_rows=events,
        source_hashes=source_hashes,
    )
    disposition = disposition_ledger.to_records()
    intervals = build_game_time_intervals_v1(ledger).to_records()
    raw_count = len(pbp_rows)
    if not (
        raw_count == len(events) == len(stats) == len(disposition)
    ):
        raise HistoricalSportsCleanError(
            f"{game_id} row reconciliation failed: raw={raw_count}, "
            f"events={len(events)}, stats={len(stats)}, "
            f"disposition={len(disposition)}"
        )
    if not episodes or not intervals:
        raise HistoricalSportsCleanError(
            f"{game_id} produced an empty episode/interval stage"
        )
    for stage_name, rows in (
        ("canonical_events", events),
        ("finalized_episodes", episodes),
        ("stat_ledger", stats),
        ("sports_row_disposition", disposition),
        ("game_time_intervals", intervals),
    ):
        if any(str(row.get("game_id")) != game_id for row in rows):
            raise HistoricalSportsCleanError(
                f"{game_id} leaked another game into {stage_name}"
            )
    if len({str(row["raw_play_id"]) for row in disposition}) != raw_count:
        raise HistoricalSportsCleanError(
            f"{game_id} disposition did not preserve every raw play identity"
        )
    return (
        {
            "canonical_events": events,
            "finalized_episodes": episodes,
            "stat_ledger": stats,
            "sports_row_disposition": disposition,
            "game_time_intervals": intervals,
        },
        {
            "canonical_events_rows_sha256": ledger.events_sha256,
            "ledger_artifact_sha256": ledger.artifact_sha256,
            "disposition_rows_sha256": disposition_ledger.rows_sha256,
            "source_row_adjudications_sha256": _canonical_sha256(
                {
                    "schema": "nflverse_source_row_adjudication_v1",
                    "rows": adjudications,
                }
            ),
        },
        len(adjudications),
    )


def _missing_source_value(value: object) -> bool:
    return value is None or (
        isinstance(value, (float, np.floating))
        and not math.isfinite(float(value))
    )


def _adjudicate_replay_rows(
    pbp_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Correct one documented nflverse administrative-row classification.

    CAR-JAX contains a weather-delay resumption notice whose source
    ``kickoff_attempt`` indicator is set although the row has no clock, source
    timestamp, teams, or live-play outcome.  Keeping that indicator would turn
    the notice into an eligible kickoff episode and manufacture a Layer-G
    event time.  The source row itself remains present in disposition and
    canonical output; only its live-action indicator is adjudicated.
    """

    result: list[dict[str, object]] = []
    adjudications: list[dict[str, object]] = []
    for raw in pbp_rows:
        row = dict(raw)
        description = row.get("desc", row.get("description"))
        normalized = (
            description.strip().lower()
            if isinstance(description, str)
            else ""
        )
        kickoff = row.get("kickoff_attempt")
        kickoff_flag = (
            isinstance(kickoff, (int, float, np.integer, np.floating))
            and not isinstance(kickoff, (bool, np.bool_))
            and math.isfinite(float(kickoff))
            and float(kickoff) == 1.0
        )
        if (
            normalized.startswith("the game has resumed.")
            and kickoff_flag
            and _missing_source_value(row.get("time_of_day"))
            and _missing_source_value(row.get("play_type"))
            and _missing_source_value(row.get("posteam"))
            and _missing_source_value(row.get("defteam"))
        ):
            row["kickoff_attempt"] = 0.0
            adjudications.append(
                {
                    "game_id": str(row["game_id"]),
                    "play_id": str(int(float(row["play_id"]))),
                    "field": "kickoff_attempt",
                    "source_value": 1,
                    "adjudicated_value": 0,
                    "reason": (
                        "administrative_game_resumption_notice_without_"
                        "live_play_or_source_time"
                    ),
                }
            )
        result.append(row)
    return result, adjudications


_STAGE_CONTRACTS = {
    "canonical_events": (
        "CanonicalGameEventV1",
        "one row per frozen nflverse PBP source row",
    ),
    "finalized_episodes": (
        "FinalizedEpisodeV1",
        "one row per finalized sports information episode",
    ),
    "stat_ledger": (
        "CumulativeStatLedgerEntryV1",
        "one cumulative snapshot per canonical source row",
    ),
    "sports_row_disposition": (
        "sports_row_disposition_v1",
        "one adjudication row per frozen nflverse PBP source row",
    ),
    "game_time_intervals": (
        "game_time_intervals_v1",
        "one finalized episode per frozen delay scenario when source time exists",
    ),
}


def _input_fingerprint(
    coverage: CandidateCoverage,
    pbp: VerifiedSource,
    participation: VerifiedSource,
    builder_code_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "builder_code_sha256": builder_code_sha256,
            "candidate_coverage_object_sha256": coverage.object_sha256,
            "pbp_manifest_sha256": pbp.manifest_logical_sha256,
            "pbp_object_sha256": pbp.object_sha256,
            "participation_manifest_sha256": (
                participation.manifest_logical_sha256
            ),
            "participation_object_sha256": participation.object_sha256,
            "reaction_access": REACTION_ACCESS,
            "schema": SCHEMA_VERSION,
        }
    )


def _verify_bundle_manifest(
    output_root: Path,
    *,
    manifest_path: Path,
    manifest_sha256: str,
    game_id: str,
    input_fingerprint: str,
) -> dict[str, Any]:
    _verify_file(manifest_path, expected_sha256=manifest_sha256)
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema")
        != "nfl_historical_sports_clean_single_game_manifest_v1"
        or manifest.get("reaction_access") != REACTION_ACCESS
        or manifest.get("game_id") != game_id
        or manifest.get("input_fingerprint") != input_fingerprint
        or manifest.get("publication_gate") != "PASS"
    ):
        raise HistoricalSportsCleanError(
            f"published single-game manifest is not reusable: {manifest_path}"
        )
    stages = manifest.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(_STAGE_CONTRACTS):
        raise HistoricalSportsCleanError(
            f"single-game stage inventory is incomplete: {game_id}"
        )
    for descriptor in stages.values():
        if not isinstance(descriptor, dict):
            raise HistoricalSportsCleanError("stage descriptor is malformed")
        object_path = _resolve_under(
            output_root,
            str(descriptor["object_path"]),
        )
        _verify_file(
            object_path,
            expected_sha256=str(descriptor["object_sha256"]),
            expected_length=int(descriptor["byte_length"]),
        )
        metadata = pq.ParquetFile(object_path).metadata
        if metadata.num_rows != int(descriptor["row_count"]):
            raise HistoricalSportsCleanError(
                f"Parquet row count drifted: {object_path}"
            )
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise HistoricalSportsCleanError("bundle counts are absent")
    raw_count = int(counts["raw_pbp_rows"])
    if not (
        raw_count
        == int(counts["canonical_event_rows"])
        == int(counts["stat_ledger_rows"])
        == int(counts["disposition_rows"])
    ):
        raise HistoricalSportsCleanError(
            f"published row reconciliation failed: {game_id}"
        )
    return manifest


def _checkpoint_path(output_root: Path, game_id: str) -> Path:
    return output_root / ".work" / "checkpoints" / f"{game_id}.json"


def _load_checkpoint(
    output_root: Path,
    *,
    game_id: str,
    input_fingerprint: str,
) -> PublishedGameBundle | None:
    checkpoint_path = _checkpoint_path(output_root, game_id)
    if not checkpoint_path.exists():
        return None
    checkpoint = _read_json(checkpoint_path)
    if (
        checkpoint.get("schema")
        != "nfl_historical_sports_clean_checkpoint_v1"
        or checkpoint.get("reaction_access") != REACTION_ACCESS
        or checkpoint.get("game_id") != game_id
        or checkpoint.get("input_fingerprint") != input_fingerprint
    ):
        return None
    manifest_path = _resolve_under(
        output_root,
        str(checkpoint["manifest_path"]),
    )
    manifest_sha256 = str(checkpoint["manifest_sha256"])
    manifest = _verify_bundle_manifest(
        output_root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        game_id=game_id,
        input_fingerprint=input_fingerprint,
    )
    counts = {key: int(value) for key, value in manifest["counts"].items()}
    counts.setdefault("source_adjudication_rows", 0)
    return PublishedGameBundle(
        game_id=game_id,
        bundle_sha256=str(manifest["bundle_sha256"]),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        counts=counts,
        reused_checkpoint=True,
        elapsed_seconds=0.0,
    )


def _publish_game(
    *,
    output_root: Path,
    game_id: str,
    pbp_rows: Sequence[Mapping[str, object]],
    participation_rows: Sequence[Mapping[str, object]],
    coverage: CandidateCoverage,
    pbp_source: VerifiedSource,
    participation_source: VerifiedSource,
    builder_code_sha256: str,
    input_fingerprint: str,
) -> PublishedGameBundle:
    started = time.perf_counter()
    reusable = _load_checkpoint(
        output_root,
        game_id=game_id,
        input_fingerprint=input_fingerprint,
    )
    if reusable is not None:
        return reusable
    source_hashes = {
        "nflverse_2025_pbp_object": pbp_source.object_sha256,
        "nflverse_2025_participation_object": (
            participation_source.object_sha256
        ),
    }
    stages, ledger_hashes, adjudication_count = _stage_records_for_game(
        game_id,
        pbp_rows,
        participation_rows,
        source_hashes,
    )
    descriptors: dict[str, dict[str, object]] = {}
    for stage_name, records in stages.items():
        schema_name, grain = _STAGE_CONTRACTS[stage_name]
        descriptors[stage_name] = _publish_parquet_stage(
            output_root,
            game_id=game_id,
            stage_name=stage_name,
            schema_name=schema_name,
            grain=grain,
            records=records,
        )
    counts = {
        "raw_pbp_rows": len(pbp_rows),
        "participation_rows": len(participation_rows),
        "canonical_event_rows": len(stages["canonical_events"]),
        "finalized_episode_rows": len(stages["finalized_episodes"]),
        "stat_ledger_rows": len(stages["stat_ledger"]),
        "disposition_rows": len(stages["sports_row_disposition"]),
        "game_time_interval_rows": len(stages["game_time_intervals"]),
        "source_adjudication_rows": adjudication_count,
    }
    manifest_material = {
        "schema": "nfl_historical_sports_clean_single_game_manifest_v1",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "X-13",
        "game_id": game_id,
        "reaction_access": REACTION_ACCESS,
        "source_scope": "sports_only",
        "claim_boundary": (
            "SPORTS CLEAN/REPLAY ONLY; no market response, reaction, "
            "latency, causality, execution, or alpha conclusion"
        ),
        "input_fingerprint": input_fingerprint,
        "inputs": {
            "candidate_coverage_manifest_file_sha256": (
                coverage.manifest_file_sha256
            ),
            "candidate_coverage_object_sha256": coverage.object_sha256,
            "candidate_selection_basis": coverage.selection_basis,
            "pbp_manifest_sha256": pbp_source.manifest_logical_sha256,
            "pbp_manifest_file_sha256": pbp_source.manifest_file_sha256,
            "pbp_object_sha256": pbp_source.object_sha256,
            "pbp_license_status": pbp_source.license_status,
            "participation_manifest_sha256": (
                participation_source.manifest_logical_sha256
            ),
            "participation_manifest_file_sha256": (
                participation_source.manifest_file_sha256
            ),
            "participation_object_sha256": (
                participation_source.object_sha256
            ),
            "participation_license_status": (
                participation_source.license_status
            ),
        },
        "builder_code_sha256": builder_code_sha256,
        "builder_hashes": dict(ledger_hashes),
        "counts": counts,
        "reconciliation": {
            "raw_rows_silently_dropped": 0,
            "raw_to_canonical": "PASS",
            "raw_to_disposition": "PASS",
            "raw_to_stat_ledger": "PASS",
            "game_identity": "PASS",
            "hash_verification": "PASS",
        },
        "stages": descriptors,
        "publication_gate": "PASS",
    }
    bundle_sha256 = _canonical_sha256(manifest_material)
    manifest = {**manifest_material, "bundle_sha256": bundle_sha256}
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    digest = manifest_sha256.removeprefix("sha256:")
    relative_manifest = (
        Path("single-game")
        / game_id
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.manifest.json"
    )
    manifest_path = _resolve_under(output_root, relative_manifest)
    observed_sha256 = _atomic_json(manifest_path, manifest)
    if observed_sha256 != manifest_sha256:
        raise HistoricalSportsCleanError("manifest publication hash drifted")
    verified = _verify_bundle_manifest(
        output_root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        game_id=game_id,
        input_fingerprint=input_fingerprint,
    )
    checkpoint = {
        "schema": "nfl_historical_sports_clean_checkpoint_v1",
        "game_id": game_id,
        "reaction_access": REACTION_ACCESS,
        "input_fingerprint": input_fingerprint,
        "bundle_sha256": bundle_sha256,
        "manifest_path": manifest_path.relative_to(output_root).as_posix(),
        "manifest_sha256": manifest_sha256,
        "counts": counts,
    }
    checkpoint_path = _checkpoint_path(output_root, game_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(checkpoint)
    temporary = checkpoint_path.parent / (
        f".{checkpoint_path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_bytes(encoded)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, checkpoint_path)
    return PublishedGameBundle(
        game_id=game_id,
        bundle_sha256=bundle_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        counts={key: int(value) for key, value in verified["counts"].items()},
        reused_checkpoint=False,
        elapsed_seconds=time.perf_counter() - started,
    )


def _publish_batch_index(
    output_root: Path,
    *,
    scope: str,
    coverage: CandidateCoverage,
    input_fingerprint: str,
    games: Sequence[PublishedGameBundle],
) -> tuple[str, Path, str, dict[str, int]]:
    ordered = tuple(sorted(games, key=lambda item: item.game_id))
    aggregate: dict[str, int] = {}
    for game in ordered:
        for name, count in game.counts.items():
            aggregate[name] = aggregate.get(name, 0) + int(count)
    material = {
        "schema": "nfl_historical_sports_clean_batch_index_v1",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "X-13",
        "scope": scope,
        "reaction_access": REACTION_ACCESS,
        "source_scope": "sports_only",
        "input_fingerprint": input_fingerprint,
        "candidate_coverage_object_sha256": coverage.object_sha256,
        "requested_game_count": len(ordered),
        "published_game_count": len(ordered),
        "failed_game_count": 0,
        "aggregate_counts": aggregate,
        "games": [
            {
                "game_id": game.game_id,
                "bundle_sha256": game.bundle_sha256,
                "manifest_path": game.manifest_path.relative_to(
                    output_root
                ).as_posix(),
                "manifest_sha256": game.manifest_sha256,
                "counts": dict(game.counts),
            }
            for game in ordered
        ],
        "publication_gate": "PASS",
        "claim_boundary": (
            "SPORTS CLEAN/REPLAY ONLY; reaction access remained CLOSED"
        ),
    }
    batch_sha256 = _canonical_sha256(material)
    index = {**material, "batch_sha256": batch_sha256}
    encoded = _canonical_bytes(index)
    index_sha256 = _sha256_bytes(encoded)
    digest = index_sha256.removeprefix("sha256:")
    relative = (
        Path("batches")
        / "manifests"
        / "sha256"
        / digest[:2]
        / f"{digest}.batch-index.json"
    )
    path = _resolve_under(output_root, relative)
    observed = _atomic_json(path, index)
    if observed != index_sha256:
        raise HistoricalSportsCleanError("batch index publication hash drifted")
    return batch_sha256, path, index_sha256, aggregate


def materialize_historical_sports_clean(
    *,
    project_root: Path,
    data_root: Path | None = None,
    output_root: Path | None = None,
    candidate_manifest_path: Path | None = None,
    game_ids: Sequence[str] | None = None,
    workers: int = 4,
    progress: Any | None = None,
) -> PublishedSportsCleanBatch:
    """Materialize one bounded game or the frozen 234-game sports scope."""

    started = time.perf_counter()
    project_root = project_root.resolve()
    data_root = (
        discover_main_checkout(project_root)
        if data_root is None
        else data_root.resolve()
    )
    output_root = (
        _resolve_under(project_root, DEFAULT_OUTPUT_ROOT)
        if output_root is None
        else output_root.resolve()
    )
    coverage = verify_candidate_coverage(
        project_root,
        candidate_manifest_path,
    )
    requested = (
        coverage.game_ids
        if game_ids is None
        else tuple(sorted(str(game_id) for game_id in game_ids))
    )
    if not requested or len(requested) != len(set(requested)):
        raise HistoricalSportsCleanError(
            "requested game IDs must be nonempty and unique"
        )
    unknown = sorted(set(requested) - set(coverage.game_ids))
    if unknown:
        raise HistoricalSportsCleanError(
            f"game is outside frozen candidate coverage: {unknown}"
        )
    if workers < 1 or workers > 4:
        raise HistoricalSportsCleanError("workers must be between 1 and 4")
    pbp_source, participation_source = verify_frozen_2025_sources(data_root)
    code_sha256 = _builder_code_sha256()
    fingerprint = _input_fingerprint(
        coverage,
        pbp_source,
        participation_source,
        code_sha256,
    )
    pbp_frame, participation_frame = _load_sports_frames(
        pbp_source,
        participation_source,
        requested,
    )
    pbp_indices = {
        str(game_id): indices
        for game_id, indices in pbp_frame.groupby(
            "game_id", sort=False
        ).indices.items()
    }
    participation_indices = {
        str(game_id): indices
        for game_id, indices in participation_frame.groupby(
            "game_id", sort=False
        ).indices.items()
    }

    def build_one(game_id: str) -> PublishedGameBundle:
        pbp_rows = _frame_records(pbp_frame.iloc[pbp_indices[game_id]])
        part_index = participation_indices.get(game_id)
        participation_rows = (
            []
            if part_index is None
            else _frame_records(participation_frame.iloc[part_index])
        )
        return _publish_game(
            output_root=output_root,
            game_id=game_id,
            pbp_rows=pbp_rows,
            participation_rows=participation_rows,
            coverage=coverage,
            pbp_source=pbp_source,
            participation_source=participation_source,
            builder_code_sha256=code_sha256,
            input_fingerprint=fingerprint,
        )

    completed: list[PublishedGameBundle] = []
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(
        max_workers=min(workers, len(requested)),
        thread_name_prefix="nfl-sports-clean",
    ) as pool:
        futures = {pool.submit(build_one, game_id): game_id for game_id in requested}
        for future in as_completed(futures):
            game_id = futures[future]
            try:
                published = future.result()
            except Exception as exc:  # publication gate reports all game failures
                failures[game_id] = f"{type(exc).__name__}: {exc}"
            else:
                completed.append(published)
                if progress is not None:
                    progress(published, len(completed), len(requested))
    if failures:
        # Single-game content-addressed bundles/checkpoints remain valid resume
        # points, but an incomplete batch index is never published.
        raise HistoricalSportsCleanError(
            "batch publication refused after per-game failures: "
            + json.dumps(failures, ensure_ascii=False, sort_keys=True)
        )
    scope = (
        "all-234-reaction-closed"
        if requested == coverage.game_ids
        else "bounded-smoke-reaction-closed"
    )
    batch_sha, index_path, index_sha, aggregate = _publish_batch_index(
        output_root,
        scope=scope,
        coverage=coverage,
        input_fingerprint=fingerprint,
        games=completed,
    )
    return PublishedSportsCleanBatch(
        scope=scope,
        game_count=len(completed),
        batch_sha256=batch_sha,
        index_path=index_path,
        index_sha256=index_sha,
        games=tuple(sorted(completed, key=lambda item: item.game_id)),
        aggregate_counts=aggregate,
        elapsed_seconds=time.perf_counter() - started,
    )


def batch_result_record(
    batch: PublishedSportsCleanBatch,
    *,
    output_root: Path,
) -> dict[str, object]:
    """Return a compact JSON-safe CLI result without reaction fields."""

    return {
        "schema": "nfl_historical_sports_clean_run_result_v1",
        "scope": batch.scope,
        "reaction_access": REACTION_ACCESS,
        "game_count": batch.game_count,
        "batch_sha256": batch.batch_sha256,
        "index_path": batch.index_path.relative_to(output_root).as_posix(),
        "index_sha256": batch.index_sha256,
        "aggregate_counts": dict(batch.aggregate_counts),
        "elapsed_seconds": round(batch.elapsed_seconds, 6),
        "reused_checkpoint_count": sum(
            int(game.reused_checkpoint) for game in batch.games
        ),
    }
