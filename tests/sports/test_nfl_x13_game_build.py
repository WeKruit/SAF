from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

import prediction_market.sports.nfl_x13_game_build as game_build_module
from prediction_market.sports.nfl_x13_game_build import (
    X13GameBuildError,
    X13GameStateBuild,
    _census_identity_material,
    _scores,
    build_x13_game_state,
    build_x13_game_state_manifest,
    publish_x13_game_state,
    verify_x13_game_state_publication,
    x13_game_state_manifest_semantic_sha256,
)
from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS
from prediction_market.sports.nfl_x13_spec import canonical_sha256


def _ledger_hashes() -> dict[str, str]:
    return {
        game_id: f"sha256:{index:064x}"
        for index, game_id in enumerate(X13_GAME_IDS, start=1)
    }


def _rewrite_checksums(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    ledger = "\n".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(root).as_posix()}"
        for path in files
    )
    (root / "checksums.sha256").write_text(ledger + "\n", encoding="ascii")


def test_final_score_comes_from_the_terminal_game_state() -> None:
    transient = SimpleNamespace(
        game_id="game",
        home_team="HOME",
        away_team="AWAY",
        game_end=False,
        pre_state=SimpleNamespace(
            home_score=10,
            away_score=3,
            terminal=False,
        ),
        post_state=SimpleNamespace(
            home_score=10,
            away_score=3,
            terminal=False,
        ),
    )
    terminal = SimpleNamespace(
        game_id="game",
        home_team="HOME",
        away_team="AWAY",
        game_end=True,
        pre_state=SimpleNamespace(
            home_score=7,
            away_score=3,
            terminal=True,
        ),
        post_state=SimpleNamespace(
            home_score=7,
            away_score=3,
            terminal=True,
        ),
    )

    assert _scores((transient, terminal)) == {"AWAY": 3, "HOME": 7}


def test_game_state_manifest_is_exact_twenty_and_content_addressed() -> None:
    manifest = build_x13_game_state_manifest(
        ledger_hashes=_ledger_hashes(),
        source_manifest_hashes=(
            f"sha256:{'a' * 64}",
            f"sha256:{'b' * 64}",
        ),
        census_sha256=f"sha256:{'c' * 64}",
        builder_version="nfl-x13-game-state-v1",
        runtime={
            "elapsed_ns": 1,
            "peak_rss_bytes": 1,
            "compute_workers": 2,
            "census_transition_count": 3_679,
            "census_latency_p50_ns": 10,
            "census_latency_p95_ns": 20,
            "census_latency_p99_ns": 30,
            "ledger_runtime_total_ns": 40,
        },
    )

    assert manifest["game_ids"] == list(X13_GAME_IDS)
    assert manifest["game_count"] == 20
    assert manifest["deterministic_two_run_hashes_equal"] is True
    assert manifest["publication_gate"] == "PASS"
    assert manifest["build_id"].startswith("sha256:")

    repeated_with_different_runtime = build_x13_game_state_manifest(
        ledger_hashes=_ledger_hashes(),
        source_manifest_hashes=(
            f"sha256:{'a' * 64}",
            f"sha256:{'b' * 64}",
        ),
        census_sha256=f"sha256:{'c' * 64}",
        builder_version="nfl-x13-game-state-v1",
        runtime={
            "elapsed_ns": 9_999,
            "peak_rss_bytes": 8_888,
            "compute_workers": 2,
            "census_transition_count": 3_679,
            "census_latency_p50_ns": 100,
            "census_latency_p95_ns": 200,
            "census_latency_p99_ns": 300,
            "ledger_runtime_total_ns": 400,
        },
    )
    assert repeated_with_different_runtime["runtime"] != manifest["runtime"]
    assert repeated_with_different_runtime["build_id"] == manifest["build_id"]

    for field, value, error_match in (
        ("compute_workers", 3, "exactly two workers"),
        ("census_transition_count", 3_680, "semantic identity"),
    ):
        tampered = dict(manifest)
        tampered["runtime"] = {**manifest["runtime"], field: value}
        with pytest.raises(X13GameBuildError, match=error_match):
            x13_game_state_manifest_semantic_sha256(tampered)

    malformed_float = dict(manifest)
    malformed_float["runtime"] = {
        **manifest["runtime"],
        "compute_workers": 2.0,
    }
    with pytest.raises(
        X13GameBuildError,
        match=r"runtime\.compute_workers must be positive",
    ):
        x13_game_state_manifest_semantic_sha256(malformed_float)

    unknown_runtime_field = dict(manifest)
    unknown_runtime_field["runtime"] = {
        **manifest["runtime"],
        "unknown": 1,
    }
    with pytest.raises(X13GameBuildError, match="exact v1 fields"):
        x13_game_state_manifest_semantic_sha256(unknown_runtime_field)

    missing = _ledger_hashes()
    missing.pop(X13_GAME_IDS[-1])
    with pytest.raises(X13GameBuildError, match="exact frozen 20"):
        build_x13_game_state_manifest(
            ledger_hashes=missing,
            source_manifest_hashes=(
                f"sha256:{'a' * 64}",
                f"sha256:{'b' * 64}",
            ),
            census_sha256=f"sha256:{'c' * 64}",
            builder_version="nfl-x13-game-state-v1",
            runtime={
                "elapsed_ns": 1,
                "peak_rss_bytes": 1,
                "compute_workers": 2,
                "census_transition_count": 3_679,
            },
        )

    with pytest.raises(
        X13GameBuildError,
        match="exact v1 fields",
    ):
        build_x13_game_state_manifest(
            ledger_hashes=_ledger_hashes(),
            source_manifest_hashes=(
                f"sha256:{'a' * 64}",
                f"sha256:{'b' * 64}",
            ),
            census_sha256=f"sha256:{'c' * 64}",
            builder_version="nfl-x13-game-state-v1",
            runtime={
                "elapsed_ns": 1,
                "peak_rss_bytes": 1,
                "compute_workers": 2,
                "census_latency_p50_ns": 10,
                "census_latency_p95_ns": 20,
                "census_latency_p99_ns": 30,
                "ledger_runtime_total_ns": 40,
            },
        )

    extra_runtime_field = {
        **manifest["runtime"],
        "unknown": 1,
    }
    with pytest.raises(X13GameBuildError, match="exact v1 fields"):
        build_x13_game_state_manifest(
            ledger_hashes=_ledger_hashes(),
            source_manifest_hashes=(
                f"sha256:{'a' * 64}",
                f"sha256:{'b' * 64}",
            ),
            census_sha256=f"sha256:{'c' * 64}",
            builder_version="nfl-x13-game-state-v1",
            runtime=extra_runtime_field,
        )


def test_census_identity_excludes_runtime_metrics_but_not_audit_content() -> None:
    first = {
        "games_total": 20,
        "completed_games": 20,
        "reducer_latency": {"p50_ns": 100, "p95_ns": 200},
    }
    second = {
        "games_total": 20,
        "completed_games": 20,
        "reducer_latency": {"p50_ns": 999, "p95_ns": 1_999},
    }

    assert canonical_sha256(_census_identity_material(first)) == canonical_sha256(
        _census_identity_material(second)
    )

    second["completed_games"] = 19
    assert canonical_sha256(_census_identity_material(first)) != canonical_sha256(
        _census_identity_material(second)
    )


def test_runtime_transition_count_must_match_census() -> None:
    runtime = {
        "elapsed_ns": 1,
        "peak_rss_bytes": 2,
        "compute_workers": 2,
        "census_transition_count": 3_679,
        "census_latency_p50_ns": 10,
        "census_latency_p95_ns": 20,
        "census_latency_p99_ns": 30,
        "ledger_runtime_total_ns": 40,
    }

    assert game_build_module._validate_runtime_against_census(
        runtime,
        {"transitions": 3_679},
    )["census_transition_count"] == 3_679
    with pytest.raises(
        X13GameBuildError,
        match="census transition count changed",
    ):
        game_build_module._validate_runtime_against_census(
            runtime,
            {"transitions": 3_680},
        )


def test_existing_publication_reuses_semantic_build_with_new_runtime_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_manifest = build_x13_game_state_manifest(
        ledger_hashes=_ledger_hashes(),
        source_manifest_hashes=(
            f"sha256:{'a' * 64}",
            f"sha256:{'b' * 64}",
        ),
        census_sha256=f"sha256:{'c' * 64}",
        builder_version="nfl-x13-game-state-v1",
        runtime={
            "elapsed_ns": 1,
            "peak_rss_bytes": 2,
            "compute_workers": 2,
            "census_transition_count": 3_679,
            "census_latency_p50_ns": 10,
            "census_latency_p95_ns": 20,
            "census_latency_p99_ns": 30,
            "ledger_runtime_total_ns": 40,
        },
    )
    later_manifest = build_x13_game_state_manifest(
        ledger_hashes=_ledger_hashes(),
        source_manifest_hashes=(
            f"sha256:{'a' * 64}",
            f"sha256:{'b' * 64}",
        ),
        census_sha256=f"sha256:{'c' * 64}",
        builder_version="nfl-x13-game-state-v1",
        runtime={
            "elapsed_ns": 9_999,
            "peak_rss_bytes": 8_888,
            "compute_workers": 2,
            "census_transition_count": 3_679,
            "census_latency_p50_ns": 100,
            "census_latency_p95_ns": 200,
            "census_latency_p99_ns": 300,
            "ledger_runtime_total_ns": 400,
        },
    )
    assert x13_game_state_manifest_semantic_sha256(first_manifest) == (
        x13_game_state_manifest_semantic_sha256(later_manifest)
    )

    target = tmp_path / str(first_manifest["build_id"]).replace(":", "-")
    target.mkdir()
    (target / "build_manifest.json").write_bytes(
        json.dumps(
            first_manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    verified_paths: list[Path] = []

    def verify_existing(path: str | Path) -> dict[str, object]:
        verified_paths.append(Path(path))
        return {
            "build_id": first_manifest["build_id"],
            "game_count": 20,
            "verified_file_count": 104,
        }

    monkeypatch.setattr(
        game_build_module,
        "verify_x13_game_state_publication",
        verify_existing,
    )
    later_build = X13GameStateBuild(
        games=MappingProxyType({game_id: object() for game_id in X13_GAME_IDS}),
        census=MappingProxyType({}),
        source=object(),  # type: ignore[arg-type]
        manifest=MappingProxyType(later_manifest),
    )

    assert publish_x13_game_state(later_build, output_root=tmp_path) == target
    assert verified_paths == [target]
    persisted = json.loads((target / "build_manifest.json").read_text("utf-8"))
    assert persisted["runtime"] == first_manifest["runtime"]
    assert persisted["runtime"] != later_manifest["runtime"]

    tampered_manifest = dict(later_manifest)
    tampered_manifest["builder_version"] = "tampered"
    tampered_build = X13GameStateBuild(
        games=later_build.games,
        census=later_build.census,
        source=later_build.source,
        manifest=MappingProxyType(tampered_manifest),
    )
    with pytest.raises(X13GameBuildError, match="semantic identity"):
        publish_x13_game_state(tampered_build, output_root=tmp_path)


@pytest.mark.skipif(
    "X13_RAW_ROOT" not in os.environ,
    reason="set X13_RAW_ROOT to run the byte-verified 20-game build",
)
def test_real_build_replays_all_twenty_twice_and_can_publish(tmp_path: Path) -> None:
    build = build_x13_game_state(
        program_root=Path(__file__).resolve().parents[2],
        raw_store_root=Path(os.environ["X13_RAW_ROOT"]),
    )

    assert tuple(build.games) == X13_GAME_IDS
    assert sum(len(game.events) for game in build.games.values()) == 3_699
    assert build.census["games_total"] == 20
    assert build.census["completed_games"] == 20
    assert build.census["fail_closed_games"] == 0
    assert build.census["deterministic"] is True
    assert build.manifest["deterministic_two_run_hashes_equal"] is True
    assert build.manifest["publication_gate"] == "PASS"
    status_counts = {
        status: sum(
            game.participation_status_counts.get(status, 0)
            for game in build.games.values()
        )
        for status in ("present_complete", "present_partial", "missing")
    }
    assert status_counts == {
        "present_complete": 3_413,
        "present_partial": 12,
        "missing": 274,
    }
    for game_id, game in build.games.items():
        assert game.game_id == game_id
        assert game.events_sha256.startswith("sha256:")
        assert game.ledger_sha256.startswith("sha256:")
        assert game.final_score_verified is True

    published = publish_x13_game_state(build, output_root=tmp_path)
    verified = verify_x13_game_state_publication(published)
    assert verified["game_count"] == 20
    assert verified["verified_file_count"] == 104

    manifest_path = published / "build_manifest.json"
    original_manifest_bytes = manifest_path.read_bytes()
    tampered_manifest = json.loads(original_manifest_bytes)
    tampered_manifest["runtime"]["compute_workers"] = 3
    manifest_path.write_text(
        json.dumps(
            tampered_manifest,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    _rewrite_checksums(published)
    with pytest.raises(X13GameBuildError, match="hash graph build manifest"):
        verify_x13_game_state_publication(published)
    manifest_path.write_bytes(original_manifest_bytes)
    _rewrite_checksums(published)

    events_path = (
        published / "games" / X13_GAME_IDS[0] / "canonical_events.json"
    )
    events = json.loads(events_path.read_text("utf-8"))
    events[0]["description"] = "tampered"
    events_path.write_text(
        json.dumps(events, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_checksums(published)

    with pytest.raises(X13GameBuildError, match="hash graph"):
        verify_x13_game_state_publication(published)
