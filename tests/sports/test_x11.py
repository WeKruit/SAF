from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market import contracts
from prediction_market.sports import x11
from prediction_market.sports.nflverse import inspect_nflverse_partition
from prediction_market.static_store import StaticStoreError


YEARS = tuple(range(2015, 2026))
PROGRAM_ROOT = Path(__file__).resolve().parents[2]


def _native_partition(
    year: int,
    *,
    observed_year: int | None = None,
    rows: list[dict[str, object]] | None = None,
) -> bytes:
    native_rows = rows or [
        {
            "play_id": 10.0,
            "game_id": f"{year}_01_A_B",
            "season": observed_year or year,
            "season_type": "REG",
            "week": 1,
            "game_date": f"{year}-09-10",
            "home_team": "B",
            "away_team": "A",
            "posteam": "B",
            "score_differential": 0.0,
            "game_seconds_remaining": 3600.0,
            "home_timeouts_remaining": 3.0,
            "away_timeouts_remaining": 3.0,
            "spread_line": 3.5,
            "home_wp": 0.58,
            "fixed_drive": 1.0,
            "fixed_drive_result": "Touchdown",
            "home_score": 24,
            "away_score": 17,
            "order_sequence": 10.0,
        },
        {
            "play_id": 20.0,
            "game_id": f"{year}_POST_C_D",
            "season": observed_year or year,
            "season_type": "POST",
            "week": 20,
            "game_date": f"{year + 1}-01-10",
            "home_team": "D",
            "away_team": "C",
            "posteam": "C",
            "score_differential": 0.0,
            "game_seconds_remaining": 3600.0,
            "home_timeouts_remaining": 3.0,
            "away_timeouts_remaining": 3.0,
            "spread_line": -1.5,
            "home_wp": 0.47,
            "fixed_drive": 1.0,
            "fixed_drive_result": "Punt",
            "home_score": 20,
            "away_score": 20,
            "order_sequence": 20.0,
        },
    ]
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(native_rows), sink)
    return sink.getvalue().to_pybytes()


def _verified(
    year: int,
    payload: bytes,
    *,
    license_ref: str = "I-018",
    license_status: str = "approved",
    source_url: str | None = None,
    source_cursor: str | None = None,
) -> SimpleNamespace:
    audit = inspect_nflverse_partition(payload, expected_year=year)
    manifest_digest = hashlib.sha256(f"manifest-{year}".encode()).hexdigest()
    manifest = SimpleNamespace(
        dataset_id="DS-NFLVERSE",
        manifest_sha256=f"sha256:{manifest_digest}",
        object_sha256=audit.object_sha256,
        schema_fingerprint=audit.schema_fingerprint,
        upstream_partition=f"season-{year}",
        coverage=f"season={year};season_type=REG,POST",
        object_kind="byte_exact_original",
        license_ref=license_ref,
        license_status=license_status,
        source_url=source_url or x11.nflverse_partition_url(year),
        source_cursor=source_cursor or x11.expected_nflverse_source_cursor(year),
    )
    record = SimpleNamespace(
        source="nflverse",
        dataset="DS-NFLVERSE",
        version=x11.X11_NFLVERSE_VERSION,
        partition=f"season-{year}",
        extension="parquet",
        manifest=manifest,
    )
    return SimpleNamespace(record=record, object_bytes=payload)


def _install_reader(
    monkeypatch: pytest.MonkeyPatch,
    payloads: dict[int, bytes],
) -> tuple[list[Path], list[Path]]:
    monkeypatch.setattr(
        x11,
        "X11_FROZEN_PARTITION_ALLOWLIST",
        {
            year: (
                inspect_nflverse_partition(
                    payloads[year],
                    expected_year=year,
                ).object_sha256,
                inspect_nflverse_partition(
                    payloads[year],
                    expected_year=year,
                ).schema_fingerprint,
            )
            for year in YEARS
        },
    )
    paths = [Path(f"/fake/season-{year}.manifest.json") for year in YEARS]
    by_path = {
        path: _verified(year, payloads[year])
        for path, year in zip(paths, YEARS, strict=True)
    }
    calls: list[Path] = []

    def reader(
        manifest_path: str | Path,
        *,
        store_root: str | Path,
        program_root: str | Path,
    ) -> SimpleNamespace:
        del store_root, program_root
        path = Path(manifest_path)
        calls.append(path)
        return by_path[path]

    monkeypatch.setattr(x11, "read_verified_static_object", reader)
    return paths, calls


def _state_frame(
    *,
    warmup_games: int = 30,
    evaluation_games: int = 12,
    evaluation_across_all_seasons: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    transition_classes = list(x11.TRANSITION_CLASSES)
    game_specs: list[tuple[int, pd.Timestamp, str]] = []
    for game_index in range(warmup_games):
        season = 2015 + min(4, game_index * 5 // warmup_games)
        within_season = game_index - (
            next(
                (
                    prior
                    for prior in range(game_index, -1, -1)
                    if 2015 + min(4, prior * 5 // warmup_games) < season
                ),
                -1,
            )
            + 1
        )
        game_specs.append(
            (
                season,
                pd.Timestamp(f"{season}-09-01T00:00:00Z")
                + pd.Timedelta(days=within_season * 3),
                "home_win" if game_index % 2 == 0 else "away_win",
            )
        )
    for game_index in range(evaluation_games):
        season = (
            2020 + min(5, game_index * 6 // evaluation_games)
            if evaluation_across_all_seasons
            else 2020
        )
        game_specs.append(
            (
                season,
                pd.Timestamp(f"{season}-09-01T00:00:00Z")
                + pd.Timedelta(days=game_index + 1),
                (
                    "tie"
                    if game_index == 3
                    else "home_win"
                    if game_index % 2 == 0
                    else "away_win"
                ),
            )
        )
    for game_index, (season, game_date, outcome) in enumerate(game_specs):
        spread_line = float((game_index % 7) - 3)
        digest = hashlib.sha256(f"season-{season}".encode()).hexdigest()
        for drive_index, transition in enumerate(transition_classes, start=1):
            remaining = float(3600 - drive_index * 500)
            differential = float((drive_index - 2) * (1 if game_index % 2 == 0 else -1))
            comparator_logit = (
                0.12 * spread_line + 0.08 * differential + 0.0003 * (3600 - remaining)
            )
            rows.append(
                {
                    "game_order": game_index,
                    "game_id": f"{season}_game_{game_index:03d}",
                    "season": season,
                    "season_type": "REG",
                    "week": game_index + 1,
                    "game_date": game_date,
                    "home_team": "HOME",
                    "away_team": "AWAY",
                    "drive_number": drive_index,
                    "play_id": float(drive_index),
                    "home_score_differential": differential,
                    "game_seconds_remaining": remaining,
                    "possession_home": float(drive_index % 2),
                    "home_timeouts_remaining": float(max(0, 3 - drive_index // 2)),
                    "away_timeouts_remaining": float(
                        max(0, 3 - (drive_index + 1) // 2)
                    ),
                    "spread_line": spread_line,
                    "home_wp": 1.0 / (1.0 + math.exp(-comparator_logit)),
                    "final_outcome": outcome,
                    "home_win": (
                        1
                        if outcome == "home_win"
                        else 0
                        if outcome == "away_win"
                        else pd.NA
                    ),
                    "next_drive_outcome": transition,
                    "manifest_sha256": f"sha256:{digest}",
                    "object_sha256": f"sha256:{digest}",
                    "schema_fingerprint": f"sha256:{digest}",
                }
            )
    frame = pd.DataFrame(rows)
    frame["home_win"] = frame["home_win"].astype("Int64")
    return frame


def _loaded_state_frame(frame: pd.DataFrame) -> x11.X11LoadedDataset:
    partitions: list[x11.X11PartitionInventory] = []
    for year in YEARS:
        digest = hashlib.sha256(f"inventory-{year}".encode()).hexdigest()
        year_frame = frame.loc[frame["season"] == year]
        partitions.append(
            x11.X11PartitionInventory(
                year=year,
                partition=f"season-{year}",
                manifest_sha256=f"sha256:{digest}",
                object_sha256=f"sha256:{digest}",
                schema_fingerprint=f"sha256:{digest}",
                rows=len(year_frame),
                games=year_frame["game_id"].nunique(),
                season_types=("REG",),
            )
        )
    inventory_without_hash = x11.X11InputInventory(
        dataset_id=x11.X11_DATASET_ID,
        source="nflverse",
        version=x11.X11_NFLVERSE_VERSION,
        years=YEARS,
        partitions=tuple(partitions),
        total_rows=len(frame),
        total_games=frame["game_id"].nunique(),
        season_types=("REG",),
        inventory_sha256="",
    )
    inventory = replace(
        inventory_without_hash,
        inventory_sha256=x11.inventory_sha256(inventory_without_hash),
    )
    return x11.X11LoadedDataset(
        inventory=inventory,
        drive_starts=frame,
        chronology_sha256=x11.chronology_sha256(frame),
        normalized_frame_sha256=x11.normalized_frame_sha256(frame),
        adapter_audit=x11.X11AdapterAudit(
            native_drives=len(frame),
            canonical_drive_starts=len(frame),
            excluded_drives_without_complete_state=0,
            games=frame["game_id"].nunique(),
            ties=frame.loc[frame["final_outcome"] == "tie", "game_id"].nunique(),
        ),
    )


def test_inventory_reads_exactly_one_verified_manifest_per_year_and_self_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = {year: _native_partition(year) for year in YEARS}
    paths, calls = _install_reader(monkeypatch, payloads)

    loaded = x11.load_x11_dataset(
        store_root=tmp_path,
        program_root=tmp_path,
        manifest_paths=reversed(paths),
    )

    assert calls == list(reversed(paths))
    assert tuple(partition.year for partition in loaded.inventory.partitions) == YEARS
    assert loaded.inventory.total_rows == 22
    assert loaded.inventory.total_games == 22
    assert loaded.inventory.season_types == ("POST", "REG")
    assert loaded.inventory.inventory_sha256 == x11.inventory_sha256(loaded.inventory)
    assert loaded.inventory.inventory_sha256.startswith("sha256:")
    with pytest.raises(FrozenInstanceError):
        loaded.inventory.total_rows = 0  # type: ignore[misc]


def test_default_discovery_ignores_append_only_legacy_license_observations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = {year: _native_partition(year) for year in YEARS}
    audits = {
        year: inspect_nflverse_partition(payloads[year], expected_year=year)
        for year in YEARS
    }
    monkeypatch.setattr(
        x11,
        "X11_FROZEN_PARTITION_ALLOWLIST",
        {
            year: (audit.object_sha256, audit.schema_fingerprint)
            for year, audit in audits.items()
        },
    )
    valid_by_path: dict[Path, SimpleNamespace] = {}
    expected_paths: list[Path] = []
    base = (
        tmp_path
        / "manifests"
        / "source=nflverse"
        / "dataset=DS-NFLVERSE"
        / f"version={x11.X11_NFLVERSE_VERSION}"
    )
    for year in YEARS:
        partition = base / f"partition=season-{year}"
        partition.mkdir(parents=True)
        legacy = partition / f"{year:064x}.manifest.json"
        governed = partition / f"{year + 100:064x}.manifest.json"
        legacy.write_text(
            json.dumps(
                {"license_ref": "O-099", "license_status": "approved"}
            ),
            encoding="utf-8",
        )
        governed.write_text(
            json.dumps(
                {
                    "dataset_id": "DS-NFLVERSE",
                    "license_ref": "I-018",
                    "license_status": "approved",
                    "object_kind": "byte_exact_original",
                    "object_sha256": audits[year].object_sha256,
                    "schema_fingerprint": audits[year].schema_fingerprint,
                    "source_cursor": x11.expected_nflverse_source_cursor(year),
                    "source_url": x11.nflverse_partition_url(year),
                    "upstream_partition": f"season-{year}",
                }
            ),
            encoding="utf-8",
        )
        valid_by_path[governed] = _verified(year, payloads[year])
        expected_paths.append(governed)

    calls: list[Path] = []

    def reader(
        manifest_path: str | Path,
        *,
        store_root: str | Path,
        program_root: str | Path,
    ) -> SimpleNamespace:
        del store_root, program_root
        path = Path(manifest_path)
        calls.append(path)
        return valid_by_path[path]

    monkeypatch.setattr(x11, "read_verified_static_object", reader)

    loaded = x11.load_x11_dataset(
        store_root=tmp_path,
        program_root=tmp_path,
    )

    assert calls == expected_paths
    assert tuple(partition.year for partition in loaded.inventory.partitions) == YEARS
    assert all(
        partition.manifest_sha256.startswith("sha256:")
        for partition in loaded.inventory.partitions
    )


@pytest.mark.parametrize(
    ("license_ref", "license_status"),
    [("O-099", "approved"), ("I-018", "research_only")],
)
def test_explicit_manifest_paths_reject_non_governed_license_binding(
    license_ref: str,
    license_status: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = {year: _native_partition(year) for year in YEARS}
    paths, _ = _install_reader(monkeypatch, payloads)
    invalid = _verified(
        2015,
        payloads[2015],
        license_ref=license_ref,
        license_status=license_status,
    )
    original_reader = x11.read_verified_static_object

    def reader(
        manifest_path: str | Path,
        *,
        store_root: str | Path,
        program_root: str | Path,
    ) -> SimpleNamespace:
        if Path(manifest_path) == paths[0]:
            return invalid
        return original_reader(
            manifest_path,
            store_root=store_root,
            program_root=program_root,
        )

    monkeypatch.setattr(x11, "read_verified_static_object", reader)

    with pytest.raises(x11.X11DataError, match="license"):
        x11.load_x11_dataset(
            store_root=tmp_path,
            program_root=tmp_path,
            manifest_paths=paths,
        )


@pytest.mark.parametrize("failure", ["source_url", "source_cursor", "object_allowlist"])
def test_explicit_manifest_paths_reject_self_consistent_noncanonical_sources(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = {year: _native_partition(year) for year in YEARS}
    paths, _ = _install_reader(monkeypatch, payloads)
    if failure == "object_allowlist":
        replacement_rows = pq.read_table(pa.BufferReader(payloads[2015])).to_pylist()
        replacement_rows[0]["home_wp"] = 0.99
        replacement = _native_partition(2015, rows=replacement_rows)
        invalid = _verified(2015, replacement)
    else:
        invalid = _verified(
            2015,
            payloads[2015],
            source_url=(
                "https://example.com/self-consistent.parquet"
                if failure == "source_url"
                else None
            ),
            source_cursor=(
                "github_release_id:58152862;asset_id:999999999"
                if failure == "source_cursor"
                else None
            ),
        )
    original_reader = x11.read_verified_static_object

    def reader(
        manifest_path: str | Path,
        *,
        store_root: str | Path,
        program_root: str | Path,
    ) -> SimpleNamespace:
        if Path(manifest_path) == paths[0]:
            return invalid
        return original_reader(
            manifest_path,
            store_root=store_root,
            program_root=program_root,
        )

    monkeypatch.setattr(x11, "read_verified_static_object", reader)

    with pytest.raises(x11.X11DataError, match="frozen source"):
        x11.load_x11_dataset(
            store_root=tmp_path,
            program_root=tmp_path,
            manifest_paths=paths,
        )


@pytest.mark.parametrize("failure", ["missing", "duplicate", "year_mismatch", "tamper"])
def test_inventory_fails_closed_on_incomplete_or_unverified_inputs(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = {year: _native_partition(year) for year in YEARS}
    paths, _ = _install_reader(monkeypatch, payloads)

    if failure == "missing":
        selected = paths[:-1]
    elif failure == "duplicate":
        selected = [*paths, paths[0]]
    elif failure == "year_mismatch":
        bad = _native_partition(2015, observed_year=2016)

        def mismatched_reader(
            manifest_path: str | Path,
            *,
            store_root: str | Path,
            program_root: str | Path,
        ) -> SimpleNamespace:
            del manifest_path, store_root, program_root
            return _verified(2015, bad)

        monkeypatch.setattr(x11, "read_verified_static_object", mismatched_reader)
        selected = paths
    else:

        def tampered_reader(
            manifest_path: str | Path,
            *,
            store_root: str | Path,
            program_root: str | Path,
        ) -> SimpleNamespace:
            del manifest_path, store_root, program_root
            raise StaticStoreError(
                "object SHA-256 does not match the governed manifest"
            )

        monkeypatch.setattr(x11, "read_verified_static_object", tampered_reader)
        selected = paths

    with pytest.raises((x11.X11DataError, StaticStoreError)):
        x11.load_x11_dataset(
            store_root=tmp_path,
            program_root=tmp_path,
            manifest_paths=selected,
        )


def test_adapter_selects_first_valid_drive_state_and_freezes_chronology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = {year: _native_partition(year) for year in YEARS}
    payloads[2015] = _native_partition(
        2015,
        rows=[
            {
                "play_id": 2.0,
                "game_id": "2015_01_A_B",
                "season": 2015,
                "season_type": "REG",
                "week": 1,
                "game_date": "2015-09-10",
                "home_team": "B",
                "away_team": "A",
                "posteam": "A",
                "score_differential": -3.0,
                "game_seconds_remaining": 3300.0,
                "home_timeouts_remaining": 3.0,
                "away_timeouts_remaining": 2.0,
                "spread_line": 3.5,
                "home_wp": 0.61,
                "fixed_drive": 2.0,
                "fixed_drive_result": "Turnover on downs",
                "home_score": 24,
                "away_score": 17,
                "order_sequence": 2.0,
            },
            {
                "play_id": 0.0,
                "game_id": "2015_01_A_B",
                "season": 2015,
                "season_type": "REG",
                "week": 1,
                "game_date": "2015-09-10",
                "home_team": "B",
                "away_team": "A",
                "posteam": None,
                "score_differential": None,
                "game_seconds_remaining": 3600.0,
                "home_timeouts_remaining": 3.0,
                "away_timeouts_remaining": 3.0,
                "spread_line": 3.5,
                "home_wp": 0.58,
                "fixed_drive": 1.0,
                "fixed_drive_result": "Touchdown",
                "home_score": 24,
                "away_score": 17,
                "order_sequence": 0.0,
            },
            {
                "play_id": 1.0,
                "game_id": "2015_01_A_B",
                "season": 2015,
                "season_type": "REG",
                "week": 1,
                "game_date": "2015-09-10",
                "home_team": "B",
                "away_team": "A",
                "posteam": "B",
                "score_differential": 0.0,
                "game_seconds_remaining": 3595.0,
                "home_timeouts_remaining": 3.0,
                "away_timeouts_remaining": 3.0,
                "spread_line": 3.5,
                "home_wp": 0.59,
                "fixed_drive": 1.0,
                "fixed_drive_result": "Touchdown",
                "home_score": 24,
                "away_score": 17,
                "order_sequence": 1.0,
            },
            {
                "play_id": 3.0,
                "game_id": "2015_POST_C_D",
                "season": 2015,
                "season_type": "POST",
                "week": 20,
                "game_date": "2016-01-10",
                "home_team": "D",
                "away_team": "C",
                "posteam": "D",
                "score_differential": 0.0,
                "game_seconds_remaining": 3600.0,
                "home_timeouts_remaining": 3.0,
                "away_timeouts_remaining": 3.0,
                "spread_line": -1.5,
                "home_wp": 0.47,
                "fixed_drive": 1.0,
                "fixed_drive_result": "Punt",
                "home_score": 20,
                "away_score": 20,
                "order_sequence": 3.0,
            },
        ],
    )
    paths, _ = _install_reader(monkeypatch, payloads)

    loaded = x11.load_x11_dataset(
        store_root=tmp_path,
        program_root=tmp_path,
        manifest_paths=paths,
    )
    frame = loaded.drive_starts
    game = frame.loc[frame["game_id"] == "2015_01_A_B"].reset_index(drop=True)

    assert list(game["drive_number"]) == [1, 2]
    assert list(game["play_id"]) == [1.0, 2.0]
    assert list(game["home_score_differential"]) == [0.0, 3.0]
    assert list(game["possession_home"]) == [1.0, 0.0]
    assert list(game["next_drive_outcome"]) == ["touchdown", "turnover"]
    assert game["final_outcome"].unique().tolist() == ["home_win"]
    assert pd.api.types.is_datetime64_any_dtype(frame["game_date"])
    assert str(frame["game_date"].dt.tz) == "UTC"
    assert frame.sort_values(
        ["game_date", "game_id", "drive_number", "play_id"],
        kind="mergesort",
    ).index.equals(frame.index)
    assert loaded.chronology_sha256.startswith("sha256:")
    assert loaded.normalized_frame_sha256 == x11.normalized_frame_sha256(frame)
    assert set(x11.GAME_STATE_FEATURES).isdisjoint(
        {
            "home_score",
            "away_score",
            "fixed_drive_result",
            "final_outcome",
            "home_win",
            "next_drive_outcome",
            "home_wp",
        }
    )
    assert list(x11.GAME_STATE_FEATURES)[-1] == "spread_prior"
    assert pd.notna(frame[list(x11.NATIVE_STATE_FEATURES)]).all().all()


def test_spread_prior_is_point_in_time_and_ignores_future_outcomes() -> None:
    original = _state_frame()
    with_prior = x11.attach_point_in_time_spread_prior(
        original,
        minimum_train_games=8,
    )
    mutated = original.copy()
    future = mutated["game_date"] >= pd.Timestamp("2020-09-08T00:00:00Z")
    mutated.loc[future & (mutated["final_outcome"] == "home_win"), "final_outcome"] = (
        "away_win"
    )
    mutated.loc[future & (mutated["home_win"] == 1), "home_win"] = 0
    mutated_prior = x11.attach_point_in_time_spread_prior(
        mutated,
        minimum_train_games=8,
    )

    available = with_prior.dropna(subset=["spread_prior"])
    assert not available.empty
    assert (available["prior_train_max_game_date"] < available["game_date"]).all()
    assert (available["prior_train_game_count"] >= 8).all()
    before_cutoff = with_prior["game_date"] < pd.Timestamp("2020-09-08T00:00:00Z")
    pd.testing.assert_series_equal(
        with_prior.loc[before_cutoff, "spread_prior"].reset_index(drop=True),
        mutated_prior.loc[before_cutoff, "spread_prior"].reset_index(drop=True),
    )
    tie_game = with_prior.loc[with_prior["final_outcome"] == "tie", "game_id"].iloc[0]
    next_game = with_prior.loc[
        with_prior["game_date"]
        > with_prior.loc[with_prior["game_id"] == tie_game, "game_date"].iloc[0]
    ].iloc[0]
    prior_training_games = original.loc[
        (original["game_date"] < next_game["game_date"])
        & (original["final_outcome"] != "tie"),
        "game_id",
    ].nunique()
    assert next_game["prior_train_game_count"] == prior_training_games
    assert with_prior["prior_method"].dropna().unique().tolist() == [
        "logistic_spread_line_strict_prior_game_dates"
    ]


def test_walk_forward_models_and_transition_distribution_are_strictly_pit() -> None:
    loaded = _loaded_state_frame(_state_frame())

    evaluation = x11.run_x11_walk_forward(
        loaded,
        program_root=PROGRAM_ROOT,
        evaluation_game_limit=10,
        minimum_prior_train_games=8,
        bootstrap_samples=40,
        minimum_valid_bootstrap_samples=20,
        confidence_level=0.90,
        gbdt_max_iter=10,
    )

    assert evaluation.result_label == "PRELIMINARY"
    assert evaluation.seed == 20260722
    assert len(evaluation.folds) == 10
    assert all(
        fold.train_max_game_date < fold.pit_cutoff
        and fold.test_game_count == 1
        and fold.train_game_count >= 30
        for fold in evaluation.folds
    )
    assert evaluation.tie_report["games_reported"] == 1
    assert evaluation.tie_report["excluded_from_binary_calibration"] is True
    assert set(evaluation.outcome_metrics) == {
        "spread_prior",
        "logistic",
        "gbdt",
        "nflfastr_home_wp",
    }
    for model_name, report in evaluation.outcome_metrics.items():
        assert {
            "brier",
            "log_loss",
            "calibration_slope",
            "calibration_intercept",
            "bootstrap_ci",
            "bootstrap_samples_requested",
            "bootstrap_samples_valid",
        } <= report.keys()
        assert report["bootstrap_samples_requested"] == 40
        assert report["bootstrap_samples_valid"] >= 20
        if model_name != "spread_prior":
            comparison = report["paired_model_minus_prior"]
            assert comparison["delta_definition"] == "model_minus_prior"
            assert set(comparison["delta_bootstrap_ci"]) == {
                "brier",
                "log_loss",
            }
    assert evaluation.model_features == {
        "logistic": x11.GAME_STATE_FEATURES,
        "gbdt": x11.GAME_STATE_FEATURES,
        "drive_transition": x11.GAME_STATE_FEATURES,
    }
    assert "home_wp" not in evaluation.model_features["logistic"]
    assert "spread_prior" in evaluation.model_features["logistic"]
    assert "spread_prior" in evaluation.model_features["gbdt"]

    transitions = evaluation.transition_predictions
    probability_columns = [f"probability_{label}" for label in x11.TRANSITION_CLASSES]
    assert np.allclose(
        transitions[probability_columns].sum(axis=1).to_numpy(),
        np.ones(len(transitions)),
        rtol=0,
        atol=1e-12,
    )
    assert transitions[probability_columns].ge(0).all().all()
    assert transitions[probability_columns].le(1).all().all()
    assert transitions["pit_cutoff_at"].str.endswith("Z").all()
    assert transitions["pit_status"].eq("PIT_UNPROVEN").all()
    assert transitions["inventory_sha256"].eq(loaded.inventory.inventory_sha256).all()
    assert (
        transitions[["manifest_sha256", "object_sha256", "schema_fingerprint"]]
        .notna()
        .all()
        .all()
    )
    assert evaluation.transition_metrics["classes"] == x11.TRANSITION_CLASSES
    assert evaluation.transition_metrics["bootstrap_samples_requested"] == 40
    assert evaluation.transition_metrics["bootstrap_samples_valid"] == 40
    assert set(evaluation.season_stability) == {"2020"}
    assert set(evaluation.season_stability["2020"]["outcome_models"]) == {
        "spread_prior",
        "logistic",
        "gbdt",
        "nflfastr_home_wp",
    }
    assert {
        "brier",
        "log_loss",
        "games",
        "observations",
    } <= evaluation.season_stability["2020"]["transition"].keys()


def test_transition_outputs_pass_the_dedicated_registry_backed_v1_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded_state_frame(_state_frame())
    calls: list[dict[str, object]] = []
    original = contracts.validate_contract_v1

    def validator(
        program_root: str | Path,
        schema_name: str,
        instance: object,
    ) -> object:
        assert Path(program_root) == PROGRAM_ROOT
        assert schema_name == "model-output/v1.schema.yaml"
        calls.append(dict(instance))  # type: ignore[arg-type]
        return original(program_root, schema_name, instance)

    monkeypatch.setattr(x11.contracts, "validate_contract_v1", validator)

    evaluation = x11.run_x11_walk_forward(
        loaded,
        program_root=PROGRAM_ROOT,
        evaluation_game_limit=10,
        minimum_prior_train_games=8,
        bootstrap_samples=40,
        minimum_valid_bootstrap_samples=20,
        confidence_level=0.90,
        gbdt_max_iter=10,
    )

    assert len(evaluation.transition_outputs) == len(
        evaluation.transition_predictions
    )
    assert len(calls) == len(evaluation.transition_outputs)
    for output in evaluation.transition_outputs:
        validated = original(
            PROGRAM_ROOT,
            "model-output/v1.schema.yaml",
            output,
        )
        assert validated.experiment_id == "X-11"
        assert validated.model_id == "MODEL-NFL-DRIVE-TRANSITION"
        assert validated.model_version == "v1"
        assert validated.game_id.startswith("game_nflverse_")
        assert validated.state_event_id.startswith("evt_")
        assert validated.transition_unit == "drive"
        assert validated.horizon == "next_state_transition"
        assert set(validated.quality_flags) == {
            "preliminary_rules",
            "source_clock_unverified",
        }
        assert validated.data_sha256 == evaluation.evaluation_input_sha256


def test_walk_forward_rejects_a_mutated_frozen_chronology() -> None:
    loaded = _loaded_state_frame(_state_frame())
    mutated_frame = loaded.drive_starts.copy()
    mutated_frame.loc[0, "game_date"] = mutated_frame.loc[
        0, "game_date"
    ] + pd.Timedelta(days=1)
    mutated = replace(loaded, drive_starts=mutated_frame)

    with pytest.raises(x11.X11DataError, match="chronology"):
        x11.run_x11_walk_forward(
            mutated,
            program_root=PROGRAM_ROOT,
            evaluation_game_limit=10,
            minimum_prior_train_games=8,
            bootstrap_samples=40,
            minimum_valid_bootstrap_samples=20,
            confidence_level=0.90,
            gbdt_max_iter=10,
        )


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("home_wp", 0.123456),
        ("next_drive_outcome", "other"),
        ("manifest_sha256", "sha256:" + "f" * 64),
        ("home_score_differential", 999.0),
    ],
)
def test_walk_forward_rejects_any_mutated_normalized_runner_input(
    column: str,
    replacement: object,
) -> None:
    loaded = _loaded_state_frame(_state_frame())
    mutated_frame = loaded.drive_starts.copy()
    mutated_frame.loc[0, column] = replacement
    mutated = replace(loaded, drive_starts=mutated_frame)

    with pytest.raises(x11.X11DataError, match="normalized"):
        x11.run_x11_walk_forward(
            mutated,
            program_root=PROGRAM_ROOT,
            evaluation_game_limit=10,
            minimum_prior_train_games=8,
            bootstrap_samples=40,
            minimum_valid_bootstrap_samples=20,
            confidence_level=0.90,
            gbdt_max_iter=10,
        )


def test_evidence_is_machine_readable_self_hashed_and_never_formal(
    tmp_path: Path,
) -> None:
    loaded = _loaded_state_frame(_state_frame())
    evaluation = x11.run_x11_walk_forward(
        loaded,
        program_root=PROGRAM_ROOT,
        evaluation_game_limit=10,
        minimum_prior_train_games=8,
        bootstrap_samples=40,
        minimum_valid_bootstrap_samples=20,
        confidence_level=0.90,
        gbdt_max_iter=10,
    )

    evidence = x11.build_x11_evidence(
        loaded,
        evaluation,
        program_root=PROGRAM_ROOT,
        execution_mode="bounded_smoke",
    )
    evidence_path = tmp_path / "x11-evidence.json"
    x11.write_x11_evidence(evidence_path, evidence)
    persisted = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert persisted == evidence
    assert evidence["result_label"] == "PRELIMINARY"
    assert evidence["is_formal_result"] is False
    assert evidence["formal_result_eligible"] is False
    assert evidence["pit_assessment"]["spread_observation_timestamp_proven"] is False
    assert "exact prior observation timestamp" in evidence["pit_assessment"]["reason"]
    assert evidence["input_inventory"]["inventory_sha256"] == (
        loaded.inventory.inventory_sha256
    )
    assert evidence["chronology_sha256"] == loaded.chronology_sha256
    assert evidence["normalized_frame_sha256"] == (
        loaded.normalized_frame_sha256
    )
    assert evidence["evaluation_input_sha256"] == (
        evaluation.evaluation_input_sha256
    )
    assert evidence["evidence_sha256"] == x11.evidence_sha256(evidence)
    assert {lock["id"] for lock in evidence["registration_locks"]} == {
        "nfl_data_manifest_and_version",
        "spread_prior_manifest",
        "pit_feature_contract",
        "model_config_and_seed",
        "bootstrap_parameters",
        "tie_policy",
        "h_split_approval",
    }
    assert all(
        lock["status"] == "registry_unresolved"
        for lock in evidence["registration_locks"]
    )
    assert evidence["walk_forward"]["seed"] == 20260722
    assert evidence["walk_forward"]["training_rule"] == (
        "complete games with game_date strictly less than test game_date"
    )
    assert evidence["outcome_evaluation"]["ties"]["excluded_from_binary_calibration"]
    assert evidence["transition_evaluation"]["state_space"] == list(
        x11.TRANSITION_CLASSES
    )
    assert evidence["transition_evaluation"]["model_outputs"]
    for output in evidence["transition_evaluation"]["model_outputs"]:
        validated = contracts.validate_contract_v1(
            PROGRAM_ROOT,
            "model-output/v1.schema.yaml",
            output,
        )
        assert validated.data_sha256 == evaluation.evaluation_input_sha256


@pytest.mark.parametrize(
    "mutation",
    [
        "outcome_prediction",
        "transition_prediction",
        "outcome_metrics",
        "transition_output",
    ],
)
def test_evidence_builder_rejects_mutated_evaluation_results(
    mutation: str,
) -> None:
    loaded = _loaded_state_frame(_state_frame())
    evaluation = x11.run_x11_walk_forward(
        loaded,
        program_root=PROGRAM_ROOT,
        evaluation_game_limit=10,
        minimum_prior_train_games=8,
        bootstrap_samples=40,
        minimum_valid_bootstrap_samples=20,
        confidence_level=0.90,
        gbdt_max_iter=10,
    )
    if mutation == "outcome_prediction":
        frame = evaluation.predictions.copy()
        frame.loc[0, "logistic_probability"] = 0.999999
        mutated = replace(evaluation, predictions=frame)
    elif mutation == "transition_prediction":
        frame = evaluation.transition_predictions.copy()
        frame.loc[0, "probability_touchdown"] = 0.999999
        mutated = replace(evaluation, transition_predictions=frame)
    elif mutation == "outcome_metrics":
        metrics = copy.deepcopy(evaluation.outcome_metrics)
        metrics["logistic"]["brier"] = 0.0
        mutated = replace(evaluation, outcome_metrics=metrics)
    else:
        outputs = copy.deepcopy(evaluation.transition_outputs)
        outputs[0]["feature_sha256"] = "sha256:" + "f" * 64
        mutated = replace(evaluation, transition_outputs=outputs)

    with pytest.raises(x11.X11DataError, match="integrity"):
        x11.build_x11_evidence(
            loaded,
            mutated,
            program_root=PROGRAM_ROOT,
            execution_mode="bounded_smoke",
        )


def test_full_evidence_requires_and_reports_every_2020_to_2025_game() -> None:
    loaded = _loaded_state_frame(
        _state_frame(
            evaluation_games=18,
            evaluation_across_all_seasons=True,
        )
    )
    evaluation = x11.run_x11_walk_forward(
        loaded,
        program_root=PROGRAM_ROOT,
        evaluation_game_limit=None,
        minimum_prior_train_games=8,
        bootstrap_samples=100,
        minimum_valid_bootstrap_samples=20,
        confidence_level=0.90,
        gbdt_max_iter=10,
    )

    evidence = x11.build_x11_evidence(
        loaded,
        evaluation,
        program_root=PROGRAM_ROOT,
        execution_mode="full",
    )

    assert evaluation.evaluation_game_limit is None
    assert len(evaluation.folds) == 18
    assert set(evaluation.season_stability) == {
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
    }
    assert evidence["execution_mode"] == "full"
    assert evidence["walk_forward"]["evaluated_games"] == 18
    assert evidence["result_label"] == "PRELIMINARY"
