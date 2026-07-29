from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from prediction_market.sports import nfl_x16_exact153 as publication
from prediction_market.sports.nfl_historical_sports_clean import VerifiedSource
from prediction_market.sports.nfl_x16_fact_extraction import (
    build_game_fact_tables as real_build_game_fact_tables,
)


def _sha_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_authority(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    assignments: list[dict[str, object]] | None = None,
    object_experiment_id: str = "X-13",
    manifest_experiment_id: str | None = None,
) -> tuple[Path, tuple[str, ...], tuple[str, ...]]:
    development = ("2025_01_AAA_BBB", "2025_02_CCC_DDD")
    holdout = ("2025_13_EEE_FFF",)
    assignment_rows = assignments or [
        {"cohort": "development", "game_id": development[0], "week": 1},
        {"cohort": "development", "game_id": development[1], "week": 2},
        {"cohort": "final_holdout", "game_id": holdout[0], "week": 13},
    ]
    object_payload = {
        "schema": "nfl_factor_expansion_registry_v2",
        "experiment_id": object_experiment_id,
        "status": "FROZEN",
        "split_lock": {
            "development": {
                "game_count": 2,
                "weeks": [1, 2],
                "access_state": "CLOSED",
                "reaction_access": False,
            },
            "final_holdout": {
                "game_count": 1,
                "weeks": [13],
                "access_state": "CLOSED",
                "reaction_access": False,
            },
            "game_assignments": assignment_rows,
            "method": "OUT_OF_TIME_BY_WEEK",
        },
    }
    object_bytes = _canonical_bytes(object_payload)
    object_sha = _sha_bytes(object_bytes)
    object_hex = object_sha.removeprefix("sha256:")
    authority_root = root / "authority"
    object_path = (
        authority_root
        / "objects"
        / "sha256"
        / object_hex[:2]
        / f"{object_hex}.json"
    )
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(object_bytes)
    manifest_payload = {
        "schema": "nfl_factor_expansion_registry_manifest_v2",
        "candidate_game_count": 3,
        "development_game_count": 2,
        "final_holdout_game_count": 1,
        "development_reaction_access": False,
        "final_holdout_reaction_access": False,
        "object_path": object_path.relative_to(authority_root).as_posix(),
        "object_sha256": object_sha,
        "byte_length": len(object_bytes),
    }
    if manifest_experiment_id is not None:
        manifest_payload["experiment_id"] = manifest_experiment_id
    manifest_bytes = _canonical_bytes(manifest_payload)
    manifest_sha = _sha_bytes(manifest_bytes)
    manifest_path = (
        authority_root
        / "manifests"
        / f"{manifest_sha.removeprefix('sha256:')}.manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(manifest_bytes)
    monkeypatch.setattr(
        publication, "EXPECTED_GOVERNANCE_MANIFEST_FILE_SHA256", manifest_sha
    )
    monkeypatch.setattr(
        publication, "EXPECTED_GOVERNANCE_OBJECT_SHA256", object_sha
    )
    monkeypatch.setattr(publication, "EXPECTED_DEVELOPMENT_GAME_COUNT", 2)
    monkeypatch.setattr(publication, "EXPECTED_HOLDOUT_GAME_COUNT", 1)
    monkeypatch.setattr(publication, "EXPECTED_DEVELOPMENT_WEEKS", (1, 2))
    monkeypatch.setattr(publication, "EXPECTED_HOLDOUT_WEEKS", (13,))
    return manifest_path, development, holdout


def _verified_source(path: Path, *, dataset_id: str) -> VerifiedSource:
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text("{}", encoding="utf-8")
    return VerifiedSource(
        dataset_id=dataset_id,
        manifest_path=manifest_path,
        manifest_logical_sha256="sha256:" + "a" * 64,
        manifest_file_sha256=_sha_bytes(manifest_path.read_bytes()),
        object_path=path,
        object_sha256=_sha_bytes(path.read_bytes()),
        byte_length=path.stat().st_size,
        license_status=(
            "approved" if dataset_id == "DS-NFLVERSE" else "research_only"
        ),
    )


def _write_sources(
    root: Path,
    development: tuple[str, ...],
) -> tuple[VerifiedSource, VerifiedSource]:
    common = {
        "home_team": "BBB",
        "away_team": "AAA",
        "qtr": 1,
        "home_timeouts_remaining": 3,
        "away_timeouts_remaining": 3,
        "play_deleted": 0,
        "goal_to_go": 0,
    }
    pbp = pd.DataFrame(
        [
            {
                **common,
                "game_id": development[0],
                "play_id": 1,
                "order_sequence": 1,
                "time": "10:00",
                "time_of_day": "2025-09-05T00:10:00Z",
                "play_type": "pass",
                "play_type_nfl": "PASS",
                "posteam": "AAA",
                "defteam": "BBB",
                "posteam_score": 0,
                "defteam_score": 0,
                "total_home_score": 0,
                "total_away_score": 6,
                "down": 1,
                "ydstogo": 10,
                "yrdln": "BBB 10",
                "end_yard_line": "BBB 0",
                "yardline_100": 10,
                "yards_gained": 10,
                "pass_attempt": 1,
                "complete_pass": 1,
                "touchdown": 1,
                "pass_touchdown": 1,
                "td_team": "AAA",
                "desc": "Touchdown.",
            },
            {
                **common,
                "game_id": development[0],
                "play_id": 2,
                "order_sequence": 2,
                "time": "10:00",
                "time_of_day": "2025-09-05T00:10:05Z",
                "play_type": "extra_point",
                "play_type_nfl": "XP_KICK",
                "posteam": "AAA",
                "defteam": "BBB",
                "posteam_score": 6,
                "defteam_score": 0,
                "total_home_score": 0,
                "total_away_score": 7,
                "down": None,
                "ydstogo": None,
                "yrdln": "BBB 15",
                "end_yard_line": "BBB 0",
                "yardline_100": 15,
                "yards_gained": 0,
                "extra_point_result": "good",
                "desc": "Extra point is good.",
            },
            {
                **common,
                "home_team": "DDD",
                "away_team": "CCC",
                "game_id": development[1],
                "play_id": 10,
                "order_sequence": 10,
                "time": "15:00",
                "time_of_day": "2025-09-12T00:00:00Z",
                "play_type": "run",
                "play_type_nfl": "RUSH",
                "posteam": "CCC",
                "defteam": "DDD",
                "posteam_score": 0,
                "defteam_score": 0,
                "total_home_score": 0,
                "total_away_score": 0,
                "down": 1,
                "ydstogo": 10,
                "yrdln": "CCC 25",
                "end_yard_line": "CCC 29",
                "yardline_100": 75,
                "yards_gained": 4,
                "rush_attempt": 1,
                "desc": "Run for four yards.",
            },
        ]
    )
    participation = pd.DataFrame(
        [
            {
                "nflverse_game_id": row["game_id"],
                "play_id": row["play_id"],
                "offense_players": "",
                "offense_names": "",
                "offense_positions": "",
                "offense_numbers": "",
                "defense_players": "",
                "defense_names": "",
                "defense_positions": "",
                "defense_numbers": "",
                "n_offense": 0,
                "n_defense": 0,
            }
            for row in pbp.to_dict("records")
        ]
    )
    pbp_path = root / "pbp.parquet"
    participation_path = root / "participation.parquet"
    pbp.to_parquet(pbp_path, index=False)
    participation.to_parquet(participation_path, index=False)
    return (
        _verified_source(pbp_path, dataset_id="DS-NFLVERSE"),
        _verified_source(
            participation_path,
            dataset_id="DS-NFLVERSE-PARTICIPATION",
        ),
    )


def _write_registry(root: Path) -> Path:
    path = root / "registry.json"
    path.write_bytes(
        _canonical_bytes(
            {
                "schema": "NFLFactorRegistryV3",
                "version": "v3-draft-single-game-review",
                "status": "DRAFT_EXPERT_REVIEW",
                "factors": [],
            }
        )
    )
    return path


def _configure_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, tuple[str, ...], tuple[str, ...], Path, dict[str, object]]:
    manifest_path, development, holdout = _write_authority(tmp_path, monkeypatch)
    sources = _write_sources(tmp_path, development)
    registry_path = _write_registry(tmp_path)
    calls: dict[str, object] = {"discover": 0, "verify": 0, "loads": []}

    def discover(project_root: Path) -> Path:
        assert project_root == tmp_path.resolve()
        calls["discover"] = int(calls["discover"]) + 1
        return tmp_path

    def verify(data_root: Path) -> tuple[VerifiedSource, VerifiedSource]:
        assert data_root == tmp_path
        calls["verify"] = int(calls["verify"]) + 1
        return sources

    monkeypatch.setattr(publication, "discover_main_checkout", discover)
    monkeypatch.setattr(publication, "verify_frozen_2025_sources", verify)
    original_loader = publication._load_game_frames

    def load_one(
        pbp_source: VerifiedSource,
        participation_source: VerifiedSource,
        game_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cast_loads = calls["loads"]
        assert isinstance(cast_loads, list)
        cast_loads.append(game_id)
        return original_loader(pbp_source, participation_source, game_id)

    monkeypatch.setattr(publication, "_load_game_frames", load_one)
    return manifest_path, development, holdout, registry_path, calls


def test_authority_derives_exact_ordered_development_and_rejects_holdout_or_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, development, holdout = _write_authority(tmp_path, monkeypatch)

    authority = publication.verify_exact153_authority(
        project_root=tmp_path,
        governance_manifest_path=manifest_path,
    )

    assert authority.development_game_ids == development
    assert authority.final_holdout_game_ids == holdout
    assert authority.manifest_file_sha256 == _sha_bytes(manifest_path.read_bytes())
    assert authority.object_sha256 == _sha_bytes(authority.object_path.read_bytes())
    authority.require_development(development[0])
    with pytest.raises(publication.NFLExact153PublicationError, match="holdout"):
        authority.require_development(holdout[0])
    with pytest.raises(publication.NFLExact153PublicationError, match="unknown"):
        authority.require_development("2025_99_UNKNOWN_GAME")


@pytest.mark.parametrize(
    ("object_experiment_id", "manifest_experiment_id"),
    [
        ("X-16", None),
        ("X-13", "X-16"),
    ],
)
def test_authority_rejects_non_x13_experiment_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    object_experiment_id: str,
    manifest_experiment_id: str | None,
) -> None:
    manifest_path, _, _ = _write_authority(
        tmp_path,
        monkeypatch,
        object_experiment_id=object_experiment_id,
        manifest_experiment_id=manifest_experiment_id,
    )

    with pytest.raises(
        publication.NFLExact153PublicationError,
        match="experiment identity",
    ):
        publication.verify_exact153_authority(
            project_root=tmp_path,
            governance_manifest_path=manifest_path,
        )


def test_authority_rejects_overlapping_development_and_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = "2025_01_AAA_BBB"
    manifest_path, _, _ = _write_authority(
        tmp_path,
        monkeypatch,
        assignments=[
            {"cohort": "development", "game_id": duplicate, "week": 1},
            {"cohort": "development", "game_id": "2025_02_CCC_DDD", "week": 2},
            {"cohort": "final_holdout", "game_id": duplicate, "week": 13},
        ],
    )

    with pytest.raises(publication.NFLExact153PublicationError, match="disjoint"):
        publication.verify_exact153_authority(
            project_root=tmp_path,
            governance_manifest_path=manifest_path,
        )


def test_workers_default_to_one_and_parallel_values_fail_closed() -> None:
    signature = inspect.signature(publication.publish_exact153_fact_batch)
    assert signature.parameters["workers"].default == 1

    with pytest.raises(
        publication.NFLExact153PublicationError,
        match="workers must equal 1",
    ):
        publication.publish_exact153_fact_batch(
            project_root=Path("/does/not/matter"),
            output_root=Path("/does/not/matter"),
            workers=2,
        )


def test_streams_one_game_at_a_time_and_verifies_frozen_sources_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, development, holdout, registry_path, calls = _configure_inputs(
        tmp_path, monkeypatch
    )

    result = publication.publish_exact153_fact_batch(
        project_root=tmp_path,
        governance_manifest_path=manifest_path,
        factor_registry_path=registry_path,
        output_root=tmp_path / "published",
    )

    assert isinstance(result, publication.PublishedExact153FactBatch)
    assert result.game_count == 2
    assert calls == {
        "discover": 1,
        "verify": 1,
        "loads": list(development),
    }
    assert not set(holdout).intersection(calls["loads"])
    index = json.loads(result.index_path.read_text(encoding="utf-8"))
    assert index["publication_gate"] == "PASS"
    assert [row["game_id"] for row in index["games"]] == list(development)
    assert len(index["games"]) == 2


def test_per_game_publication_preserves_td_pat_ids_and_fixed_empty_schemas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _, _, registry_path, _ = _configure_inputs(
        tmp_path, monkeypatch
    )
    output_root = tmp_path / "published"

    result = publication.publish_exact153_fact_batch(
        project_root=tmp_path,
        governance_manifest_path=manifest_path,
        factor_registry_path=registry_path,
        output_root=output_root,
    )

    game_manifest = json.loads(
        result.games[0].manifest_path.read_text(encoding="utf-8")
    )
    descriptors = {
        row["name"]: row for row in game_manifest["tables"]
    }
    events = pd.read_parquet(
        output_root / descriptors["canonical_factor_events"]["object_path"]
    )
    assert events["atomic_information_episode_id"].tolist() == [
        "2025_01_AAA_BBB:episode:1",
        "2025_01_AAA_BBB:episode:2",
    ]
    assert events["score_sequence_id"].nunique() == 1
    assert events["score_sequence_id"].notna().all()
    for name in (
        "factor_event_players",
        "player_availability_events",
        "injury_evidence",
        "factor_hits",
        "factor_coverage_audit",
    ):
        frame = pd.read_parquet(output_root / descriptors[name]["object_path"])
        assert frame.empty
        assert tuple(frame.columns) == publication.TABLE_SCHEMAS[name]
        assert descriptors[name]["schema_columns"] == list(
            publication.TABLE_SCHEMAS[name]
        )


def test_fixed_player_schema_preserves_nonempty_participation_unit(
    tmp_path: Path,
) -> None:
    pbp = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "play_id": 1,
                "order_sequence": 1,
                "home_team": "BBB",
                "away_team": "AAA",
                "qtr": 1,
                "time": "15:00",
                "time_of_day": "2025-09-05T00:00:00Z",
                "play_type": "run",
                "play_type_nfl": "RUSH",
                "posteam": "AAA",
                "defteam": "BBB",
                "total_home_score": 0,
                "total_away_score": 0,
                "rush_attempt": 1,
                "desc": "Run.",
            }
        ]
    )
    participation = pd.DataFrame(
        [
            {
                "nflverse_game_id": "2025_01_AAA_BBB",
                "play_id": 1,
                "offense_players": "00-rb",
                "offense_names": "Runner",
                "offense_positions": "RB",
                "offense_numbers": "20",
                "defense_players": "00-lb",
                "defense_names": "Linebacker",
                "defense_positions": "LB",
                "defense_numbers": "50",
                "n_offense": 1,
                "n_defense": 1,
            }
        ]
    )
    tables = real_build_game_fact_tables(
        pbp,
        participation,
        factor_registry={"factors": []},
        pbp_source_sha256="sha256:" + "a" * 64,
        participation_source_sha256="sha256:" + "b" * 64,
    )

    normalized = publication._normalize_tables(tables)
    empty = publication._ordered_frame(
        "factor_event_players",
        pd.DataFrame(),
    )

    assert tuple(normalized.players.columns) == publication.TABLE_SCHEMAS[
        "factor_event_players"
    ]
    assert set(normalized.players["unit"]) == {"OFFENSE", "DEFENSE"}
    assert tuple(map(str, empty.dtypes)) == tuple(
        map(str, normalized.players.dtypes)
    )
    empty_descriptor = publication._publish_table(
        output_root=tmp_path,
        game_id="empty",
        name="factor_event_players",
        frame=empty,
    )
    populated_descriptor = publication._publish_table(
        output_root=tmp_path,
        game_id="populated",
        name="factor_event_players",
        frame=normalized.players,
    )
    empty_schema = pq.read_schema(
        tmp_path / empty_descriptor["object_path"]
    ).remove_metadata()
    populated_schema = pq.read_schema(
        tmp_path / populated_descriptor["object_path"]
    ).remove_metadata()
    assert empty_schema == populated_schema


def test_zero_orphan_duplicate_and_silent_loss_audits_are_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _, _, registry_path, _ = _configure_inputs(
        tmp_path, monkeypatch
    )

    result = publication.publish_exact153_fact_batch(
        project_root=tmp_path,
        governance_manifest_path=manifest_path,
        factor_registry_path=registry_path,
        output_root=tmp_path / "published",
    )

    for game in result.games:
        manifest = json.loads(game.manifest_path.read_text(encoding="utf-8"))
        audit = manifest["referential_integrity"]
        assert audit["pbp_duplicate_key_rows"] == 0
        assert audit["participation_duplicate_key_rows"] == 0
        assert audit["participation_orphan_rows"] == 0
        assert audit["raw_rows_silently_dropped"] == 0
        assert audit["event_fk_orphan_rows"] == 0
        assert audit["publication_gate"] == "PASS"


@pytest.mark.parametrize("failure", ["duplicate_pbp", "participation_orphan"])
def test_source_ri_failures_leave_no_pass_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    manifest_path, _, _, registry_path, _ = _configure_inputs(
        tmp_path, monkeypatch
    )
    original_verify = publication.verify_frozen_2025_sources
    pbp_source, participation_source = original_verify(tmp_path)
    if failure == "duplicate_pbp":
        pbp = pd.read_parquet(pbp_source.object_path)
        pd.concat([pbp, pbp.iloc[[0]]], ignore_index=True).to_parquet(
            pbp_source.object_path, index=False
        )
    else:
        participation = pd.read_parquet(participation_source.object_path)
        orphan = participation.iloc[[0]].copy()
        orphan["play_id"] = 999
        pd.concat([participation, orphan], ignore_index=True).to_parquet(
            participation_source.object_path, index=False
        )
    updated = (
        _verified_source(pbp_source.object_path, dataset_id="DS-NFLVERSE"),
        _verified_source(
            participation_source.object_path,
            dataset_id="DS-NFLVERSE-PARTICIPATION",
        ),
    )
    monkeypatch.setattr(
        publication,
        "verify_frozen_2025_sources",
        lambda _: updated,
    )
    output_root = tmp_path / "published"

    with pytest.raises(publication.NFLExact153PublicationError):
        publication.publish_exact153_fact_batch(
            project_root=tmp_path,
            governance_manifest_path=manifest_path,
            factor_registry_path=registry_path,
            output_root=output_root,
        )

    assert not list(output_root.rglob("*.batch-index.json"))


def test_silent_loss_from_extractor_blocks_batch_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _, _, registry_path, _ = _configure_inputs(
        tmp_path, monkeypatch
    )

    def drop_reconciliation(*args: object, **kwargs: object) -> object:
        tables = real_build_game_fact_tables(*args, **kwargs)
        return replace(tables, reconciliation=tables.reconciliation.iloc[:-1])

    monkeypatch.setattr(
        publication, "build_game_fact_tables", drop_reconciliation
    )
    output_root = tmp_path / "published"

    with pytest.raises(
        publication.NFLExact153PublicationError,
        match="reconciliation",
    ):
        publication.publish_exact153_fact_batch(
            project_root=tmp_path,
            governance_manifest_path=manifest_path,
            factor_registry_path=registry_path,
            output_root=output_root,
        )

    assert not list(output_root.rglob("*.batch-index.json"))


def test_child_event_fk_failure_blocks_batch_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _, _, registry_path, _ = _configure_inputs(
        tmp_path, monkeypatch
    )

    def add_orphan_tag(*args: object, **kwargs: object) -> object:
        tables = real_build_game_fact_tables(*args, **kwargs)
        orphan = pd.DataFrame(
            [
                {
                    "game_id": tables.events.iloc[0]["game_id"],
                    "event_id": "missing-event",
                    "play_id": "missing-play",
                    "tag": "ORPHAN",
                    "pbp_source_sha256": "sha256:" + "a" * 64,
                }
            ]
        )
        return replace(
            tables,
            tags=pd.concat([tables.tags, orphan], ignore_index=True),
        )

    monkeypatch.setattr(publication, "build_game_fact_tables", add_orphan_tag)
    output_root = tmp_path / "published"

    with pytest.raises(
        publication.NFLExact153PublicationError,
        match="foreign key",
    ):
        publication.publish_exact153_fact_batch(
            project_root=tmp_path,
            governance_manifest_path=manifest_path,
            factor_registry_path=registry_path,
            output_root=output_root,
        )

    assert not list(output_root.rglob("*.batch-index.json"))


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_event_id",
        "permuted_reconciliation_episode_id",
        "child_event_play_mismatch",
    ],
)
def test_exact_parent_child_mappings_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    manifest_path, _, _, registry_path, _ = _configure_inputs(
        tmp_path, monkeypatch
    )

    def corrupt_mapping(*args: object, **kwargs: object) -> object:
        tables = real_build_game_fact_tables(*args, **kwargs)
        if len(tables.events) < 2:
            return tables
        events = tables.events.copy()
        reconciliation = tables.reconciliation.copy()
        tags = tables.tags.copy()
        if corruption == "duplicate_event_id":
            first_event_id = str(events.iloc[0]["event_id"])
            second_event_id = str(events.iloc[1]["event_id"])
            events.loc[events.index[1], "event_id"] = first_event_id
            reconciliation.loc[
                reconciliation["event_id"].eq(second_event_id), "event_id"
            ] = first_event_id
            tags.loc[tags["event_id"].eq(second_event_id), "event_id"] = (
                first_event_id
            )
        elif corruption == "permuted_reconciliation_episode_id":
            reconciliation["atomic_information_episode_id"] = list(
                reversed(reconciliation["atomic_information_episode_id"].tolist())
            )
        else:
            first_event_id = str(events.iloc[0]["event_id"])
            second_play_id = str(events.iloc[1]["raw_play_id"])
            target = tags.index[tags["event_id"].eq(first_event_id)][0]
            tags.loc[target, "play_id"] = second_play_id
        return replace(
            tables,
            events=events,
            reconciliation=reconciliation,
            tags=tags,
        )

    monkeypatch.setattr(publication, "build_game_fact_tables", corrupt_mapping)
    output_root = tmp_path / "published"

    with pytest.raises(
        publication.NFLExact153PublicationError,
        match="duplicate|mapping|link",
    ):
        publication.publish_exact153_fact_batch(
            project_root=tmp_path,
            governance_manifest_path=manifest_path,
            factor_registry_path=registry_path,
            output_root=output_root,
        )

    assert not list(output_root.rglob("*.batch-index.json"))


def test_exact_set_mismatch_blocks_pass_batch_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _, _, _, _ = _configure_inputs(tmp_path, monkeypatch)
    authority = publication.verify_exact153_authority(
        project_root=tmp_path,
        governance_manifest_path=manifest_path,
    )
    output_root = tmp_path / "published"

    with pytest.raises(
        publication.NFLExact153PublicationError,
        match="exact development set",
    ):
        publication._publish_batch_index(
            output_root=output_root,
            authority=authority,
            games=(),
            source_bindings={},
            registry_binding={},
            builder_code_sha256="sha256:" + "b" * 64,
        )

    assert not list(output_root.rglob("*.batch-index.json"))


def test_semantic_batch_identity_ignores_physical_bundle_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, development, _ = _write_authority(tmp_path, monkeypatch)
    authority = publication.verify_exact153_authority(
        project_root=tmp_path,
        governance_manifest_path=manifest_path,
    )
    source_bindings = {"pbp": {"logical_manifest_sha256": "sha256:source"}}
    registry_binding = {"semantic_sha256": "sha256:registry"}
    games = tuple(
        publication.PublishedGameFactBundle(
            game_id=game_id,
            bundle_sha256="sha256:" + "a" * 64,
            manifest_path=tmp_path / f"{game_id}.manifest.json",
            manifest_sha256="sha256:" + "b" * 64,
            counts={"canonical_factor_events_rows": 1},
            table_semantic_sha256={
                "canonical_factor_events": (
                    "sha256:" + ("c" if index == 0 else "d") * 64
                )
            },
        )
        for index, game_id in enumerate(development)
    )
    physically_different_games = tuple(
        replace(
            game,
            bundle_sha256="sha256:" + "e" * 64,
            manifest_sha256="sha256:" + "f" * 64,
        )
        for game in games
    )
    semantically_different_games = (
        replace(
            physically_different_games[0],
            table_semantic_sha256={
                "canonical_factor_events": "sha256:" + "9" * 64
            },
        ),
        *physically_different_games[1:],
    )

    def publish_variant(
        descriptors: tuple[publication.PublishedGameFactBundle, ...],
        builder_hash_character: str,
    ) -> publication.PublishedExact153FactBatch:
        return publication._publish_batch_index(
            output_root=tmp_path,
            authority=authority,
            games=descriptors,
            source_bindings=source_bindings,
            registry_binding=registry_binding,
            builder_code_sha256="sha256:" + builder_hash_character * 64,
        )

    first = publish_variant(games, "1")
    physically_different = publish_variant(physically_different_games, "2")
    semantically_different = publish_variant(
        semantically_different_games,
        "2",
    )

    assert first.semantic_batch_sha256 == (
        physically_different.semantic_batch_sha256
    )
    assert first.batch_sha256 != physically_different.batch_sha256
    assert first.semantic_batch_sha256 != (
        semantically_different.semantic_batch_sha256
    )


def test_bounded_rerun_has_identical_semantic_hashes_and_batch_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _, _, registry_path, _ = _configure_inputs(
        tmp_path, monkeypatch
    )
    output_root = tmp_path / "published"
    first = publication.publish_exact153_fact_batch(
        project_root=tmp_path,
        governance_manifest_path=manifest_path,
        factor_registry_path=registry_path,
        output_root=output_root,
    )
    second = publication.publish_exact153_fact_batch(
        project_root=tmp_path,
        governance_manifest_path=manifest_path,
        factor_registry_path=registry_path,
        output_root=output_root,
    )

    assert first.batch_sha256 == second.batch_sha256
    assert first.semantic_batch_sha256 == second.semantic_batch_sha256
    assert first.index_sha256 == second.index_sha256
    assert first.index_path == second.index_path
    assert [game.bundle_sha256 for game in first.games] == [
        game.bundle_sha256 for game in second.games
    ]
    for first_game, second_game in zip(first.games, second.games, strict=True):
        assert first_game.table_semantic_sha256 == (
            second_game.table_semantic_sha256
        )


def test_published_contracts_use_x13_experiment_identity_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _, _, registry_path, _ = _configure_inputs(
        tmp_path, monkeypatch
    )
    output_root = tmp_path / "published"

    batch = publication.publish_exact153_fact_batch(
        project_root=tmp_path,
        governance_manifest_path=manifest_path,
        factor_registry_path=registry_path,
        output_root=output_root,
    )

    assert publication.EXPERIMENT_ID == "X-13"
    assert publication.BUILDER_VERSION == "nfl_x13_exact153_fact_publication_v1"
    batch_payload = json.loads(batch.index_path.read_text(encoding="utf-8"))
    assert batch_payload["experiment_id"] == "X-13"
    assert batch_payload["schema"] == "nfl_x13_exact153_fact_batch_index_v1"
    assert batch_payload["builder_version"] == publication.BUILDER_VERSION
    assert batch_payload["authority"]["experiment_id"] == "X-13"
    for game in batch.games:
        game_material = game.manifest_path.read_text(encoding="utf-8")
        game_payload = json.loads(game_material)
        assert game_payload["experiment_id"] == "X-13"
        assert (
            game_payload["schema"]
            == "nfl_x13_exact153_single_game_fact_manifest_v1"
        )
        assert game_payload["builder_version"] == publication.BUILDER_VERSION
        assert game_payload["authority"]["experiment_id"] == "X-13"
        assert "x16" not in game_material.casefold()
    assert "x16" not in batch.index_path.read_text(encoding="utf-8").casefold()
