from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from prediction_market.sports import (
    nfl_factor_lab_run_index as run_index_module,
)
from prediction_market.sports.nfl_factor_lab_bundles import (
    build_cross_game_bundle,
    build_single_game_bundle,
)
from prediction_market.sports.nfl_factor_lab_run_index import (
    FactorLabRunIndexError,
    build_factor_lab_run_index,
    factor_lab_materializer_code_lock_paths,
    factor_lab_materializer_code_lock_sha256,
    verify_factor_lab_run_index,
)
from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
GAME = "2025_14_DAL_DET"


def _single_frames(game_id: str = GAME) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    grains = {
        "sports_row_disposition": {
            "raw_play_id": "1",
            "canonical_play_id": "1",
            "episode_id": "e1",
        },
        "canonical_events": {"play_id": "1"},
        "finalized_episodes": {"episode_id": "e1"},
        "stat_ledger": {"through_play_id": "1"},
        "pit_features": {"play_id": "1"},
        "contract_inventory": {
            "logical_market_id": "winner",
            "outcome": "DET",
        },
        "market_observations": {
            "observation_id": "o1",
            "logical_market_id": "winner",
            "outcome": "DET",
        },
        "market_cleaning_audit": {
            "audit_row_id": "m1",
            "observation_id": "o1",
            "included": True,
        },
        "game_time_intervals": {
            "interval_id": "g1",
            "episode_id": "e1",
            "delay_seconds": 0,
        },
        "market_time_intervals": {
            "interval_id": "m1",
            "observation_id": "o1",
        },
        "source_timeline": {
            "timeline_id": "t1",
            "interval_id": "g1",
            "layer": "G",
            "delay_seconds": 0,
        },
        "timeline_audit": {
            "timeline_audit_id": "ta1",
            "timeline_id": "t1",
        },
        "association_attrition": {
            "association_id": "a1",
            "episode_id": "e1",
            "logical_market_id": "winner",
            "outcome": "DET",
        },
        "reaction_paths": {
            "episode_id": "e1",
            "logical_market_id": "winner",
            "outcome": "DET",
            "venue": "polymarket",
            "market_family": "moneyline",
            "horizon_seconds": 30,
        },
        "factor_observations": {
            "factor_id": "NFL.EVENT.PASS",
            "episode_id": "e1",
            "logical_market_id": "winner",
            "outcome": "DET",
            "venue": "polymarket",
            "market_family": "moneyline",
            "horizon_seconds": 30,
        },
        "factor_exclusion_audit": {"audit_id": "f1"},
        "single_game_factor_results": {
            "factor_id": "NFL.EVENT.PASS",
            "market_family": "moneyline",
            "horizon_seconds": 30,
        },
    }
    for name, fields in grains.items():
        result[name] = pd.DataFrame([{"game_id": game_id, **fields}])
    return result


def _cross_frames() -> dict[str, pd.DataFrame]:
    return {
        "cross_game_factor_results": pd.DataFrame(
            [
                {
                    "factor_id": "NFL.EVENT.PASS",
                    "market_family": "moneyline",
                    "horizon_seconds": 30,
                }
            ]
        ),
        "game_contribution": pd.DataFrame(
            [
                {
                    "factor_id": "NFL.EVENT.PASS",
                    "market_family": "moneyline",
                    "horizon_seconds": 30,
                    "game_id": game_id,
                }
                for game_id in X13_GAME_IDS
            ]
        ),
        "leave_one_game_out": pd.DataFrame(
            [
                {
                    "factor_id": "NFL.EVENT.PASS",
                    "market_family": "moneyline",
                    "horizon_seconds": 30,
                    "game_id": game_id,
                }
                for game_id in X13_GAME_IDS
            ]
        ),
        "venue_consistency": pd.DataFrame(
            [
                {
                    "factor_id": "NFL.EVENT.PASS",
                    "market_family": "moneyline",
                    "horizon_seconds": 30,
                    "venue": "polymarket",
                }
            ]
        ),
        "expert_review_queue": pd.DataFrame(
            [
                {
                    "factor_id": "NFL.EVENT.PASS",
                    "market_family": "moneyline",
                    "horizon_seconds": 30,
                    "decision": "PENDING",
                }
            ]
        ),
    }


def _write_materializer_lock_files(project_root: Path) -> None:
    for relative in factor_lab_materializer_code_lock_paths():
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))


@pytest.mark.parametrize(
    "relative",
    (
        "src/prediction_market/program_audit.py",
        "src/prediction_market/sports/nfl_x13_spec.py",
        "src/prediction_market/sports/nfl_factor_lab_types.py",
        "src/prediction_market/sports/nfl_factor_lab_artifacts.py",
        "src/prediction_market/sports/nfl_x13_capture.py",
        "src/prediction_market/sports/nfl_x13_market.py",
        "src/prediction_market/sports/nfl_x13_universe.py",
    ),
)
def test_authoritative_code_lock_binds_formal_pipeline_dependencies(
    tmp_path: Path,
    relative: str,
) -> None:
    for code_path in factor_lab_materializer_code_lock_paths():
        path = tmp_path / code_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(code_path.encode("utf-8"))
    before = factor_lab_materializer_code_lock_sha256(tmp_path)

    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"changed")

    assert factor_lab_materializer_code_lock_sha256(tmp_path) != before


def _published_bundles(
    tmp_path: Path,
    *,
    code_lock_sha256: str,
):
    singles = tuple(
        build_single_game_bundle(
            game_id=game_id,
            stage_frames=_single_frames(game_id),
            source_hashes=(HASH_A,),
            factor_registry_sha256=HASH_B,
            code_lock_sha256=code_lock_sha256,
            output_root=tmp_path / "single",
        )
        for game_id in X13_GAME_IDS
    )
    cross = build_cross_game_bundle(
        single_game_manifests=tuple(
            single.manifest_path for single in singles
        ),
        cross_game_frames=_cross_frames(),
        batch_spec_sha256=HASH_A,
        factor_registry_sha256=HASH_B,
        code_lock_sha256=code_lock_sha256,
        output_root=tmp_path / "cross",
    )
    return singles, cross


def _build_run_index(
    tmp_path: Path,
    *,
    singles,
    cross,
):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source")
    return build_factor_lab_run_index(
        single_game_manifests=tuple(
            single.manifest_path for single in singles
        ),
        cross_game_manifest=cross.manifest_path,
        source_artifacts={"raw_sports": source},
        project_root=tmp_path,
        s3_bucket="prediction-market-data",
        s3_prefix="prediction-market",
        output_path=tmp_path / "nfl_factor_lab_run_index_v2.json",
    )


def test_run_index_build_rejects_an_internally_consistent_invented_code_lock(
    tmp_path: Path,
) -> None:
    _write_materializer_lock_files(tmp_path)
    singles, cross = _published_bundles(
        tmp_path,
        code_lock_sha256=HASH_A,
    )

    with pytest.raises(
        FactorLabRunIndexError,
        match="authoritative current code lock",
    ):
        _build_run_index(tmp_path, singles=singles, cross=cross)


def test_run_index_verify_rejects_an_internally_consistent_invented_code_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_materializer_lock_files(tmp_path)
    singles, cross = _published_bundles(
        tmp_path,
        code_lock_sha256=HASH_A,
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            run_index_module,
            "factor_lab_materializer_code_lock_sha256",
            lambda _project_root: HASH_A,
            raising=False,
        )
        built = _build_run_index(tmp_path, singles=singles, cross=cross)

    with pytest.raises(
        FactorLabRunIndexError,
        match="authoritative current code lock",
    ):
        verify_factor_lab_run_index(
            built.path,
            project_root=tmp_path,
        )


def test_run_index_exposes_every_verified_stage_and_source(
    tmp_path: Path,
) -> None:
    _write_materializer_lock_files(tmp_path)
    authoritative_code_lock = factor_lab_materializer_code_lock_sha256(
        tmp_path
    )
    singles = tuple(
        build_single_game_bundle(
            game_id=game_id,
            stage_frames=_single_frames(game_id),
            source_hashes=(HASH_A,),
            factor_registry_sha256=HASH_B,
            code_lock_sha256=authoritative_code_lock,
            output_root=tmp_path / "single",
        )
        for game_id in X13_GAME_IDS
    )
    cross = build_cross_game_bundle(
        single_game_manifests=tuple(
            single.manifest_path for single in singles
        ),
        cross_game_frames=_cross_frames(),
        batch_spec_sha256=HASH_A,
        factor_registry_sha256=HASH_B,
        code_lock_sha256=authoritative_code_lock,
        output_root=tmp_path / "cross",
    )
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source")
    index_path = tmp_path / "nfl_factor_lab_run_index_v2.json"

    built = build_factor_lab_run_index(
        single_game_manifests=tuple(
            single.manifest_path for single in singles
        ),
        cross_game_manifest=cross.manifest_path,
        source_artifacts={"raw_sports": source},
        project_root=tmp_path,
        s3_bucket="prediction-market-data",
        s3_prefix="prediction-market",
        output_path=index_path,
    )
    verified = verify_factor_lab_run_index(index_path, project_root=tmp_path)

    assert built == verified
    assert verified.game_ids == X13_GAME_IDS
    assert len(verified.stage_manifest) == (
        len(X13_GAME_IDS) * len(_single_frames()) + len(_cross_frames())
    )
    assert verified.holdout_reaction_accessed is False
    assert verified.code_lock_sha256 == authoritative_code_lock

    original_index = index_path.read_bytes()
    mismatched = json.loads(original_index)
    mismatched["code_lock_sha256"] = HASH_B
    index_path.write_text(
        json.dumps(
            mismatched,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FactorLabRunIndexError, match="code lock"):
        verify_factor_lab_run_index(index_path, project_root=tmp_path)
    index_path.write_bytes(original_index)

    source.write_bytes(b"tampered")
    with pytest.raises(FactorLabRunIndexError, match="source"):
        verify_factor_lab_run_index(index_path, project_root=tmp_path)
