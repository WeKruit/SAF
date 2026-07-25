from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market.sports.nfl_x13_batch import (
    association_arrow_schema,
)
from prediction_market.sports.nfl_x13_spec import canonical_sha256


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _episode() -> dict[str, object]:
    return {
        "audit_only": False,
        "blocked_kick": False,
        "contaminated": False,
        "episode_id": "episode-1",
        "episode_type": "score_turnover",
        "game_id": "2025_01_BAL_BUF",
        "nullified": False,
        "penalty_status": "none",
        "play_ids": ["10"],
        "post_state": None,
        "pre_state": {
            "away_score": 7,
            "away_team": "BAL",
            "away_timeouts_remaining": 3,
            "clock_seconds_remaining": 120,
            "defense_team": "BUF",
            "direction": "left",
            "distance": 4,
            "down": 2,
            "drive_id": "4",
            "game_seconds_remaining": 1020,
            "goal_to_go": False,
            "home_score": 10,
            "home_team": "BUF",
            "home_timeouts_remaining": 2,
            "offense_team": "BAL",
            "possession_team": "BAL",
            "quarter": 3,
            "terminal": False,
            "yardline_100": 34,
        },
        "review_result": "none",
        "score_points": 7,
        "scoring_team": "BAL",
        "source_time_end_utc": "2025-09-08T01:00:01.000Z",
        "source_time_start_utc": "2025-09-08T01:00:00.000Z",
        "turnover": True,
    }


def _contract() -> dict[str, object]:
    return {
        "analysis_eligible": True,
        "comparator": "eq",
        "condition_id": None,
        "contract_id": "kalshi:winner",
        "dependency_game_ids": ["2025_01_BAL_BUF"],
        "family": "moneyline",
        "kind": "primitive",
        "line": None,
        "logical_market_id": "kalshi:2025_01_BAL_BUF:winner",
        "measure": "winner",
        "native_outcomes": ["BAL", "BUF"],
        "native_subject": "BAL at BUF",
        "outcome_coverage": {"BAL": "available", "BUF": "available"},
        "outcomes": ["BAL", "BUF"],
        "period": "game",
        "raw_contract_ids": ["KXNFLGAME-BAL"],
        "rule_sha256": f"sha256:{'1' * 64}",
        "subject": "2025_01_BAL_BUF",
        "venue": "kalshi",
        "venue_market_id": "KXNFLGAME",
    }


def _association(
    *,
    validity_status: str = "OBSERVED",
    contaminated: bool = False,
    order_ambiguous: bool = False,
    horizon: int = 1,
) -> dict[str, object]:
    return {
        "game_id": "2025_01_BAL_BUF",
        "episode_id": "episode-1",
        "episode_type": "score_turnover",
        "contract_id": "kalshi:winner",
        "logical_market_id": "kalshi:2025_01_BAL_BUF:winner",
        "venue": "kalshi",
        "family": "moneyline",
        "outcome": "BAL",
        "source_time_start_utc": "2025-09-08T01:00:00.000000Z",
        "source_time_end_utc": "2025-09-08T01:00:01.000000Z",
        "delay_scenario_seconds": 0,
        "horizon_seconds": horizon,
        "pre_event_actual_trade": "0.5000",
        "first_post_event_trade": "0.6000",
        "vwap": "0.5900",
        "signed_price_change": "0.1000",
        "maximum_excursion": "0.1200",
        "net_change_60s": "0.0800",
        "trade_count": 2,
        "volume": "10.00",
        "staleness_seconds": "0",
        "overshoot_candidate": False,
        "reversal_candidate": False,
        "two_venue_direction_consistency": "SINGLE_VENUE",
        "order_ambiguous": order_ambiguous,
        "contaminated": contaminated,
        "validity_status": validity_status,
    }


def _fixture_batch(tmp_path: Path) -> tuple[Path, str]:
    batch = tmp_path / (
        "sha256-3ddafb13fe58faa3caf01fa720a60f6c2efc4e87d9977b129b3c3824b43c1d85"
    )
    (batch / "games").mkdir(parents=True)
    (batch / "data/associations").mkdir(parents=True)
    (batch / "reports").mkdir()
    game = {
        "associations": [{"intentionally": "skipped"} for _ in range(100)],
        "contracts": [_contract()],
        "episodes": [_episode()],
        "game_id": "2025_01_BAL_BUF",
        "observations": [
            {
                "kind": "trade",
                "observation_id": f"sha256:{'2' * 64}",
                "outcome": "BAL",
                "price": "0.5000",
                "primary_path_eligible": True,
                "provenance": "observed",
                "raw_market_id": "KXNFLGAME-BAL",
                "source_end": "2025-09-08T00:59:51.000000Z",
                "source_start": "2025-09-08T00:59:50.000000Z",
                "venue": "kalshi",
            },
            {
                "kind": "trade",
                "observation_id": f"sha256:{'3' * 64}",
                "outcome": "BAL",
                "price": "0.6000",
                "primary_path_eligible": True,
                "provenance": "observed",
                "raw_market_id": "KXNFLGAME-BAL",
                "source_end": "2025-09-08T01:00:02.000000Z",
                "source_start": "2025-09-08T01:00:01.000000Z",
                "venue": "kalshi",
            },
        ],
    }
    (batch / "games/2025_01_BAL_BUF.json").write_bytes(_canonical(game))

    rows = [
        _association(horizon=1),
        _association(horizon=2),
        _association(contaminated=True),
        _association(order_ambiguous=True),
        _association(validity_status="NO_POST_TRADE"),
    ]
    association_path = batch / "data/associations/part-00000.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=association_arrow_schema()),
        association_path,
        compression="zstd",
    )
    association_sha = hashlib.sha256(association_path.read_bytes()).hexdigest()
    manifest = {
        "auxiliary_artifacts": [
            {
                "byte_length": association_path.stat().st_size,
                "media_type": "application/vnd.apache.parquet",
                "relative_path": "data/associations/part-00000.parquet",
                "role": "association_candidate_table",
                "row_count": 5,
                "schema": "nfl_x13_auxiliary_artifact_v1",
                "schema_fingerprint": (
                    "sha256:eff4dbfc1745fdc0e3af6c961a5104474782fc0f5ea2741d610b8ab3a86ae491"
                ),
                "sha256": f"sha256:{association_sha}",
            }
        ],
        "batch_id": (
            "sha256:3ddafb13fe58faa3caf01fa720a60f6c2efc4e87d9977b129b3c3824b43c1d85"
        ),
    }
    (batch / "batch_manifest.json").write_bytes(_canonical(manifest))
    semantic_orientation_id = canonical_sha256(
        [
            "2025_01_BAL_BUF",
            "moneyline",
            "game",
            "winner",
            "eq",
            None,
            "BAL",
        ]
    )
    audit = {
        "actual_cross_venue_orientation_count": 154,
        "approved_cross_venue_orientation_count": 154,
        "approved_cross_venue_semantic_key_sha256s": [
            semantic_orientation_id
        ],
        "schema": "nfl_x13_contract_semantic_overlap_audit_v1",
        "status": "PASS",
        "unapproved_cross_venue_orientation_count": 0,
    }
    (batch / "reports/contract-semantic-overlap-audit-v1.json").write_bytes(
        _canonical(audit)
    )
    return batch, semantic_orientation_id


def test_frozen_observed_row_count_is_exact() -> None:
    from prediction_market.sports.nfl_x13_correlation_cache import (
        X13_EXPECTED_OBSERVED_ROW_COUNT,
    )

    assert X13_EXPECTED_OBSERVED_ROW_COUNT == 333_340


def test_tampered_batch_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market.sports import nfl_x13_correlation_cache as module

    batch, _ = _fixture_batch(tmp_path)
    output = tmp_path / "output"

    def reject(_path: Path, _snapshot: Path) -> Path:
        raise ValueError("checksum mismatch")

    assert not hasattr(module, "verify_published_batch")
    monkeypatch.setattr(module, "_snapshot_and_verify_frozen_batch", reject)
    with pytest.raises(module.X13CorrelationCacheError, match="strict verification"):
        module.build_x13_correlation_cache(batch, output_root=output)
    assert not list(output.glob("sha256-*"))


def test_dimension_extraction_skips_large_association_payload(
    tmp_path: Path,
) -> None:
    from prediction_market.sports.nfl_x13_correlation_cache import (
        _extract_game_dimensions,
    )

    batch, _ = _fixture_batch(tmp_path)
    episodes, contracts = _extract_game_dimensions(
        batch / "games/2025_01_BAL_BUF.json"
    )

    assert len(episodes) == 1
    assert episodes[0]["episode_superclass"] == "score"
    assert episodes[0]["episode_is_turnover"] is True
    assert episodes[0]["signal_event_eligible"] is False
    assert episodes[0]["signal_exclusion_reason"] == "score_turnover_excluded"
    assert episodes[0]["pre_score_margin_home"] == 3
    assert contracts[0]["contract_subject"] == "2025_01_BAL_BUF"


def test_materialization_retains_all_cell_validity_and_is_deterministic(
    tmp_path: Path,
) -> None:
    from prediction_market.sports.nfl_x13_correlation_cache import (
        _materialize_verified_cache,
    )

    batch, semantic_orientation_id = _fixture_batch(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    result_a = _materialize_verified_cache(
        batch, first, expected_observed_rows=2
    )
    result_b = _materialize_verified_cache(
        batch, second, expected_observed_rows=2
    )

    assert result_a["cache_parquet_sha256"] == result_b[
        "cache_parquet_sha256"
    ]
    cache = first / "x13_reaction_observed_cache_v1.parquet"
    connection = duckdb.connect()
    rows = connection.execute(
        """
        SELECT
          count(*),
          count(*) FILTER (
            WHERE validity_status = 'OBSERVED'
              AND NOT contaminated
              AND NOT order_ambiguous
          ),
          min(semantic_orientation_id),
          bool_and(cross_venue_eligible),
          min(pre_game_seconds_remaining),
          min(pre_quarter),
          min(pre_score_margin_home),
          min(pre_possession_team),
          min(pre_down),
          min(pre_distance),
          min(pre_yardline_100),
          bool_and(NOT pre_goal_to_go)
          , min(pre_trade_age_status)
          , min(pre_trade_age_seconds)
          , bool_and(pre_trade_fresh_le_60s)
          , bool_and(NOT signal_candidate_eligible)
          , min(scoring_team)
          , min(pre_offense_team)
          , min(pre_defense_team)
        FROM read_parquet(?)
        """,
        [str(cache)],
    ).fetchone()
    assert rows[:2] == (
        5,
        2,
    )
    assert rows[2:] == (
        semantic_orientation_id,
        True,
        1020,
        3,
        3,
        "BAL",
        2,
        4,
        34,
        True,
        "NOT_APPLICABLE_INVALID_CELL",
        9.0,
        True,
        True,
        "BAL",
        "BAL",
        "BUF",
    )
    validity = connection.execute(
        """
        SELECT validity_status, contaminated, order_ambiguous, count(*)
        FROM read_parquet(?)
        GROUP BY ALL
        ORDER BY validity_status, contaminated, order_ambiguous
        """,
        [str(cache)],
    ).fetchall()
    assert validity == [
        ("NO_POST_TRADE", False, False, 1),
        ("OBSERVED", False, False, 2),
        ("OBSERVED", False, True, 1),
        ("OBSERVED", True, False, 1),
    ]
    names = {field.name for field in pq.read_schema(cache)}
    assert not names & {
        "alpha",
        "p_value",
        "t_stat",
        "effect_estimate",
        "execution_price",
    }


def test_referential_integrity_failure_is_closed(tmp_path: Path) -> None:
    from prediction_market.sports.nfl_x13_correlation_cache import (
        X13CorrelationCacheError,
        _materialize_verified_cache,
    )

    batch, _ = _fixture_batch(tmp_path)
    table_path = batch / "data/associations/part-00000.parquet"
    table = pq.read_table(table_path)
    values = table.to_pylist()
    values[0]["episode_id"] = "missing"
    pq.write_table(
        pa.Table.from_pylist(values, schema=association_arrow_schema()),
        table_path,
        compression="zstd",
    )
    target = tmp_path / "target"
    target.mkdir()

    with pytest.raises(X13CorrelationCacheError, match="referential integrity"):
        _materialize_verified_cache(batch, target, expected_observed_rows=2)


def test_coverage_reports_parameter_repetition_without_inference(
    tmp_path: Path,
) -> None:
    from prediction_market.sports.nfl_x13_correlation_cache import (
        _materialize_verified_cache,
    )

    batch, _ = _fixture_batch(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    _materialize_verified_cache(batch, target, expected_observed_rows=2)

    coverage = json.loads(
        (target / "x13_reaction_coverage_matrix_v1.json").read_text()
    )
    assert coverage["claim_scope"] == "DESCRIPTIVE_CACHE_ONLY"
    assert coverage["observed_row_count"] == 2
    assert coverage["independent_sports_event_count"] == 1
    assert coverage["game_episode_venue_unit_count"] == 1
    assert coverage["parameter_repetition_row_count"] == 1
    assert coverage["inference_result_count"] == 0
    assert coverage["pre_trade_age_known_row_count"] == 2
    assert coverage["signal_candidate_eligible_row_count"] == 0
    matrix = pq.read_table(
        target / "x13_reaction_coverage_matrix_v1.parquet"
    ).to_pylist()
    assert len(matrix) == 4
    assert sum(row["row_count"] for row in matrix) == 5
    assert {
        (
            row["validity_status"],
            row["order_ambiguous"],
            row["contaminated"],
            row["row_count"],
        )
        for row in matrix
    } == {
        ("NO_POST_TRADE", False, False, 1),
        ("OBSERVED", False, False, 2),
        ("OBSERVED", False, True, 1),
        ("OBSERVED", True, False, 1),
    }


def _fixture_bundle(tmp_path: Path, name: str) -> Path:
    from prediction_market.sports.nfl_x13_correlation_cache import (
        _build_cache_manifest,
        _publish_bundle,
        _materialize_verified_cache,
    )

    batch, _ = _fixture_batch(tmp_path)
    stage = tmp_path / f"{name}-stage"
    stage.mkdir()
    identities = _materialize_verified_cache(
        batch, stage, expected_observed_rows=2
    )
    manifest = _build_cache_manifest(
        identities,
        batch_id=(
            "sha256:3ddafb13fe58faa3caf01fa720a60f6c2efc4e87d9977b129b3c3824b43c1d85"
        ),
        candidate_row_count=5,
        cross_venue_orientation_count=1,
    )
    return _publish_bundle(
        stage,
        output_root=tmp_path / name,
        manifest=manifest,
    )


def test_verifier_recomputes_identity_and_every_descriptor(
    tmp_path: Path,
) -> None:
    from prediction_market.sports.nfl_x13_correlation_cache import (
        X13CorrelationCacheError,
        _verify_cache_directory,
    )

    target = _fixture_bundle(tmp_path, "valid")
    manifest = _verify_cache_directory(target, require_frozen=False)
    assert manifest["observed_row_count"] == 2
    assert set(manifest["artifacts"]) == {
        "coverage_matrix_json",
        "coverage_matrix_parquet",
        "observed_cache",
    }
    assert not (target / "runtime-report-v1.json").exists()

    manifest["artifacts"]["observed_cache"]["row_count"] = 1
    (target / "manifest.json").write_bytes(_canonical(manifest))
    files = sorted(
        path
        for path in target.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (target / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in files
        ),
        "ascii",
    )
    with pytest.raises(
        X13CorrelationCacheError, match="row_count|cache_id"
    ):
        _verify_cache_directory(target, require_frozen=False)


def test_minimal_manifest_and_rehashed_ledger_is_rejected(
    tmp_path: Path,
) -> None:
    from prediction_market.sports.nfl_x13_correlation_cache import (
        X13CorrelationCacheError,
        _verify_cache_directory,
    )

    target = tmp_path / f"sha256-{'4' * 64}"
    target.mkdir()
    (target / "manifest.json").write_bytes(
        _canonical(
            {
                "cache_id": f"sha256:{'4' * 64}",
                "schema": "nfl_x13_correlation_cache_manifest_v1",
            }
        )
    )
    digest = hashlib.sha256((target / "manifest.json").read_bytes()).hexdigest()
    (target / "checksums.sha256").write_text(
        f"{digest}  manifest.json\n", "ascii"
    )
    with pytest.raises(X13CorrelationCacheError, match="manifest"):
        _verify_cache_directory(target, require_frozen=False)


def test_fully_rehashed_semantic_tamper_is_not_an_approved_frozen_cache(
    tmp_path: Path,
) -> None:
    from prediction_market.sports.nfl_x13_correlation_cache import (
        X13_APPROVED_CACHE_ID,
        X13CorrelationCacheError,
        _verify_cache_directory,
    )

    assert X13_APPROVED_CACHE_ID == (
        "sha256:d6e98965c0985a4f4769480f8d40bdabac1375e5f6b3bd66bc7bf654d02529c5"
    )
    target = _fixture_bundle(tmp_path, "semantic-tamper")
    observed_path = target / "x13_reaction_observed_cache_v1.parquet"
    observed = pq.read_table(observed_path).to_pylist()
    observed[0]["signed_price_change"] = "0.1010"
    pq.write_table(
        pa.Table.from_pylist(
            observed,
            schema=pq.read_schema(observed_path),
        ),
        observed_path,
        compression="zstd",
    )

    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    descriptor = manifest["artifacts"]["observed_cache"]
    descriptor["byte_length"] = observed_path.stat().st_size
    descriptor["sha256"] = (
        f"sha256:{hashlib.sha256(observed_path.read_bytes()).hexdigest()}"
    )
    manifest.pop("cache_id")
    manifest["cache_id"] = canonical_sha256(manifest)
    tampered = target.with_name(manifest["cache_id"].replace(":", "-"))
    target.rename(tampered)
    (tampered / "manifest.json").write_bytes(_canonical(manifest))
    files = sorted(
        path
        for path in tampered.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (tampered / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in files
        ),
        "ascii",
    )

    assert (
        _verify_cache_directory(tampered, require_frozen=False)["cache_id"]
        == manifest["cache_id"]
    )
    with pytest.raises(
        X13CorrelationCacheError, match="approved cache identity"
    ):
        _verify_cache_directory(tampered, require_frozen=True)


def test_input_snapshot_copy_rejects_symlink_and_digest_mismatch(
    tmp_path: Path,
) -> None:
    from prediction_market.sports.nfl_x13_correlation_cache import (
        X13CorrelationCacheError,
        _copy_bound_input,
    )

    root = tmp_path / "source"
    root.mkdir()
    source = root / "data.bin"
    source.write_bytes(b"frozen")
    symlink = root / "link.bin"
    symlink.symlink_to(source)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(X13CorrelationCacheError, match="symlink|regular"):
            _copy_bound_input(
                root_fd,
                PurePosixPath("link.bin"),
                tmp_path / "link-copy.bin",
                expected_sha256=hashlib.sha256(b"frozen").hexdigest(),
            )
        with pytest.raises(X13CorrelationCacheError, match="digest"):
            _copy_bound_input(
                root_fd,
                PurePosixPath("data.bin"),
                tmp_path / "bad-copy.bin",
                expected_sha256="0" * 64,
            )
    finally:
        os.close(root_fd)


def test_failed_stage_verification_never_exposes_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prediction_market.sports import nfl_x13_correlation_cache as module

    batch, _ = _fixture_batch(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    identities = module._materialize_verified_cache(
        batch, stage, expected_observed_rows=2
    )
    manifest = module._build_cache_manifest(
        identities,
        batch_id=module.X13_FROZEN_BATCH_ID,
        candidate_row_count=5,
        cross_venue_orientation_count=1,
    )
    expected_target = (
        tmp_path / "output" / manifest["cache_id"].replace(":", "-")
    )

    def reject(_path: Path, *, require_frozen: bool = True):
        raise module.X13CorrelationCacheError("stage invalid")

    monkeypatch.setattr(module, "_verify_cache_directory", reject)
    with pytest.raises(module.X13CorrelationCacheError, match="stage invalid"):
        module._publish_bundle(
            stage, output_root=tmp_path / "output", manifest=manifest
        )
    assert not expected_target.exists()


def test_two_builds_have_identical_bundle_bytes(tmp_path: Path) -> None:
    first = _fixture_bundle(tmp_path / "run-a", "output")
    second = _fixture_bundle(tmp_path / "run-b", "output")

    def identities(root: Path) -> dict[str, str]:
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.iterdir()
            if path.is_file()
        }

    assert first.name == second.name
    assert identities(first) == identities(second)
