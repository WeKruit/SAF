from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import test_nfl_x15_oof_publication as publication_support
from prediction_market.research.nfl_x15_model_selection import (
    FrozenDevelopmentAuthority,
    STAGE_B_CANDIDATE_SUITE,
)
from prediction_market.research.nfl_x15_models import X15ModelRun
from prediction_market.research.nfl_x15_oof_publication import (
    PROBABILITY_DESIGN_MATRIX,
    X15OOFPublicationError,
    load_verified_x15_oof_selection_batch,
    publish_x15_oof_batch,
    publish_x15_oof_shard,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/research/select_nfl_x15_stage_b_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "select_nfl_x15_stage_b_v2",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


BATCH_FILE_SHA256 = "sha256:" + "a" * 64
BATCH_SHA256 = "sha256:" + "b" * 64
AUTHORITY_SHA256 = "sha256:" + "c" * 64
FOLD_BY_WEEK = {
    3: "fold_01",
    4: "fold_01",
    5: "fold_02",
    6: "fold_02",
    7: "fold_03",
    8: "fold_03",
    9: "fold_04",
    10: "fold_04",
    11: "fold_05",
    12: "fold_05",
}


def _authority_games() -> tuple[tuple[str, int, str, str], ...]:
    return tuple(
        sorted(
            (
                game_id,
                week,
                f"2025-09-{position % 28 + 1:02d}T12:00:00Z",
                "sha256:" + f"{position + 1:064x}",
            )
            for position, (game_id, week) in enumerate(
                (
                    *(
                        (f"training-{index:03d}", index % 2 + 1)
                        for index in range(30)
                    ),
                    *(
                        (
                            f"validation-{index:03d}",
                            index % 10 + 3,
                        )
                        for index in range(123)
                    ),
                )
            )
        )
    )


def _batch_document() -> dict[str, object]:
    authority_games = _authority_games()
    return {
        "schema": "nfl_x15_oof_batch_index_v3",
        "selection_ready": True,
        "publication_status": "SELECTION_READY",
        "expected_execution_cell_count": 55,
        "observed_execution_cell_count": 55,
        "missing_execution_cell_count": 0,
        "batch_sha256": BATCH_SHA256,
        "cohort_authority_sha256": AUTHORITY_SHA256,
        "authority_metadata_sha256": (
            runner._canonical_sha256(authority_games)
        ),
        "authority_development_games": authority_games,
        "holdout_reaction_accessed": False,
    }


def _verified_authority():
    games = _authority_games()
    return FrozenDevelopmentAuthority(
        cohort_authority_sha256=AUTHORITY_SHA256,
        development_games=games,
        metadata_sha256=runner._canonical_sha256(games),
    )


def _projection_run(
    candidate: tuple[str, str],
) -> X15ModelRun:
    game_weeks = {
        game_id: week for game_id, week, _, _ in _authority_games()
    }
    fold_windows = {
        "fold_01": ((1, 2), (3, 4)),
        "fold_02": (tuple(range(1, 5)), (5, 6)),
        "fold_03": (tuple(range(1, 7)), (7, 8)),
        "fold_04": (tuple(range(1, 9)), (9, 10)),
        "fold_05": (tuple(range(1, 11)), (11, 12)),
    }
    rows: list[dict[str, object]] = []
    for index in range(123):
        week = index % 10 + 3
        fold_id = FOLD_BY_WEEK[week]
        train_weeks, validation_weeks = fold_windows[fold_id]
        training_ids = tuple(
            sorted(
                game_id
                for game_id, game_week in game_weeks.items()
                if game_week in train_weeks
            )
        )
        validation_ids = tuple(
            sorted(
                game_id
                for game_id, game_week in game_weeks.items()
                if game_week in validation_weeks
            )
        )
        for model_id, feature_block_id in (
            ("b0_empirical_v1", "D0"),
            candidate,
        ):
            rows.append(
                {
                    "game_id": f"validation-{index:03d}",
                    "nfl_week": week,
                    "fold_id": fold_id,
                    "training_game_ids": training_ids,
                    "validation_game_ids": validation_ids,
                    "model_id": model_id,
                    "feature_block_id": feature_block_id,
                    "venue": "polymarket",
                    "training_venue": "polymarket",
                    "calibration_venue": "polymarket",
                    "cohort_authority_sha256": AUTHORITY_SHA256,
                }
            )
    run_config = {
        "schema_version": "HistoricalTradesOnlyProbabilityPanelV2",
        "survival_probability_contract": (
            "DISCRETE_INTERVAL_SURVIVAL_PRODUCT_V1"
        ),
        "model_ids": ("b0_empirical_v1", candidate[0]),
        "feature_block_ids": ("D0", candidate[1]),
        "source_batch_manifest_path": (
            "batches/manifests/sha256/aa/"
            + "a" * 64
            + ".batch-index.json"
        ),
        "source_batch_manifest_sha256": BATCH_FILE_SHA256,
        "source_batch_sha256": BATCH_SHA256,
        "cohort_authority_sha256": AUTHORITY_SHA256,
    }
    return X15ModelRun(
        oof_predictions=pd.DataFrame(rows),
        conditional_quantiles=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        support_audit=pd.DataFrame(),
        weight_audit=pd.DataFrame(),
        run_config_sha256=runner._canonical_sha256(run_config),
        run_config=run_config,
    )


def _selection_result(*, winner: tuple[str, str] | None):
    audit = pd.DataFrame(
        [
            {
                "candidate_model_id": model_id,
                "candidate_feature_block_id": feature_block_id,
                "final_winner": (
                    winner == (model_id, feature_block_id)
                ),
            }
            for model_id, feature_block_id in STAGE_B_CANDIDATE_SUITE
        ]
    )
    winner_result = (
        None
        if winner is None
        else SimpleNamespace(
            spec=SimpleNamespace(
                candidate_model_id=winner[0],
                candidate_feature_block_id=winner[1],
            ),
            integrated_mean_improvement=0.04,
            integrated_ci_low=0.01,
            integrated_ci_high=0.07,
            anchor_game_count=40,
            anchor_mean_improvement=0.02,
        )
    )
    return SimpleNamespace(
        decision_status=(
            "NO_MODEL_ADVANCE" if winner is None else "MODEL_ADVANCE"
        ),
        winner=winner_result,
        candidate_audit=audit,
        best_integrated_mean_improvement=(
            None if winner is None else 0.04
        ),
        best_standard_error=None if winner is None else 0.01,
        one_se_threshold=None if winner is None else 0.03,
        cohort_authority_sha256=AUTHORITY_SHA256,
        shared_run_config_sha256="sha256:" + "e" * 64,
        suite_run_config_sha256="sha256:" + "f" * 64,
        candidate_audit_sha256=runner._canonical_sha256(
            audit.to_dict("records")
        ),
        schema_version="HistoricalTradesOnlyProbabilityPanelV2",
        survival_probability_contract=(
            "DISCRETE_INTERVAL_SURVIVAL_PRODUCT_V1"
        ),
        winner_rule_sha256="sha256:" + "2" * 64,
    )


def _install_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch_path: Path,
    runs: tuple[X15ModelRun, ...],
    selection_result,
) -> list[tuple[str, str]]:
    monkeypatch.setattr(
        runner,
        "load_verified_x15_oof_selection_batch",
        lambda **kwargs: (
            _batch_document(),
            batch_path,
            BATCH_FILE_SHA256,
            _verified_authority(),
        ),
    )
    calls: list[tuple[str, str]] = []

    def load_projection(**kwargs):
        candidate = (
            kwargs["candidate_model_id"],
            kwargs["candidate_feature_block_id"],
        )
        calls.append(candidate)
        return runs[STAGE_B_CANDIDATE_SUITE.index(candidate)]

    monkeypatch.setattr(
        runner,
        "load_x15_oof_selection_projection",
        load_projection,
    )
    monkeypatch.setattr(
        runner,
        "select_stage_b_v2_winner",
        lambda model_runs, *, authority: (
            selection_result
            if len(model_runs) == 8 and authority.game_count == 153
            else pytest.fail("runner did not pass the full suite/authority")
        ),
    )
    monkeypatch.setattr(
        runner,
        "verify_stage_b_v2_selection_result",
        lambda selection: (
            selection
            if selection is selection_result
            else pytest.fail("runner verified a different selection result")
        ),
    )
    return calls


@pytest.mark.parametrize(
    "winner",
    [None, ("regularized_logistic_v1", "D1")],
)
def test_runner_loads_all_projections_and_atomically_publishes_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winner: tuple[str, str] | None,
) -> None:
    batch_path = tmp_path / ("a" * 64 + ".batch-index.json")
    batch_path.write_text("{}\n", encoding="utf-8")
    runs = tuple(
        _projection_run(candidate)
        for candidate in STAGE_B_CANDIDATE_SUITE
    )
    calls = _install_fixtures(
        monkeypatch,
        batch_path=batch_path,
        runs=runs,
        selection_result=_selection_result(winner=winner),
    )

    first = runner.run_stage_b_v2_selection(
        batch_manifest_path=batch_path,
        batch_manifest_file_sha256=BATCH_FILE_SHA256,
        oof_output_root=tmp_path,
        selection_output_root=tmp_path / "selection",
    )
    second = runner.run_stage_b_v2_selection(
        batch_manifest_path=batch_path,
        batch_manifest_file_sha256=BATCH_FILE_SHA256,
        oof_output_root=tmp_path,
        selection_output_root=tmp_path / "selection",
    )

    assert calls == list(STAGE_B_CANDIDATE_SUITE) * 2
    assert first == second
    payload = first.manifest_path.read_bytes()
    assert first.manifest_sha256 == (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )
    assert first.manifest_path.name == (
        first.manifest_sha256.removeprefix("sha256:")
        + ".stage-b-v2-selection.json"
    )
    document = json.loads(payload)
    assert document["input_oof_batch"] == {
        "manifest_path": str(batch_path),
        "manifest_file_sha256": BATCH_FILE_SHA256,
        "batch_sha256": BATCH_SHA256,
    }
    assert len(document["candidate_run_config_sha256s"]) == 8
    assert document["decision_status"] == (
        "NO_MODEL_ADVANCE" if winner is None else "MODEL_ADVANCE"
    )
    assert document["winner"] is None if winner is None else (
        document["winner"]["model_id"],
        document["winner"]["feature_block_id"],
    ) == winner
    assert document["selection_result_audit_sha256"] == (
        runner._canonical_sha256(document["candidate_audit"])
    )
    assert document["holdout_reaction_accessed"] is False


def test_runner_rejects_unbound_batch_file_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_path = tmp_path / ("a" * 64 + ".batch-index.json")
    batch_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "load_verified_x15_oof_selection_batch",
        lambda **kwargs: (
            _batch_document(),
            batch_path,
            BATCH_FILE_SHA256,
            _verified_authority(),
        ),
    )

    with pytest.raises(runner.StageBSelectionRunnerError, match="file SHA"):
        runner.run_stage_b_v2_selection(
            batch_manifest_path=batch_path,
            batch_manifest_file_sha256="sha256:" + "9" * 64,
            oof_output_root=tmp_path,
            selection_output_root=tmp_path / "selection",
        )


def test_runner_rejects_partial_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_path = tmp_path / "partial.batch-index.json"
    monkeypatch.setattr(
        runner,
        "load_verified_x15_oof_selection_batch",
        lambda **kwargs: (_ for _ in ()).throw(
            X15OOFPublicationError(
                "selection projection requires a verified exact-55 batch"
            )
        ),
    )

    with pytest.raises(
        runner.StageBSelectionRunnerError, match="exact-55"
    ):
        runner.run_stage_b_v2_selection(
            batch_manifest_path=batch_path,
            batch_manifest_file_sha256=BATCH_FILE_SHA256,
            oof_output_root=tmp_path,
            selection_output_root=tmp_path / "selection",
        )


@pytest.mark.parametrize("failure", ["missing", "hash_mismatch"])
def test_runner_propagates_public_batch_authority_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    batch_path = tmp_path / ("a" * 64 + ".batch-index.json")
    batch_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "load_verified_x15_oof_selection_batch",
        lambda **kwargs: (_ for _ in ()).throw(
            X15OOFPublicationError(
                "batch authority_development_games are missing"
                if failure == "missing"
                else "batch authority metadata SHA-256 mismatch"
            )
        ),
    )

    with pytest.raises(
        runner.StageBSelectionRunnerError,
        match="authority_development_games|authority metadata SHA-256",
    ):
        runner.run_stage_b_v2_selection(
            batch_manifest_path=batch_path,
            batch_manifest_file_sha256=BATCH_FILE_SHA256,
            oof_output_root=tmp_path,
            selection_output_root=tmp_path / "selection",
        )


def test_runner_uses_only_public_verified_batch_loader() -> None:
    source = inspect.getsource(runner)

    assert "_read_selection_ready_batch" not in source
    assert "def _authority_from_batch" not in source
    assert "def _validate_projection" not in source
    assert "EXPECTED_DEVELOPMENT_GAME_COUNT" not in source
    assert "EXPECTED_OOF_VALIDATION_GAME_COUNT" not in source
    assert "EXPECTED_TRAINING_ONLY_GAME_COUNT" not in source
    assert "authority.game_weeks" not in source


@pytest.fixture(scope="module")
def published_selection_batch(tmp_path_factory):
    output_root = tmp_path_factory.mktemp("stage-b-v4-oof")
    authority = publication_support._authority()
    shards = tuple(
        publish_x15_oof_shard(
            model_run=publication_support._with_nonselection_rows(
                publication_support._fold_run(
                    fold_id,
                    authority=authority,
                    model_id=model_id,
                    feature_block_id=feature_block_id,
                )
            ),
            authority=authority,
            cohort_mapping_sha256=publication_support.MAPPING_SHA,
            output_root=output_root,
        )
        for fold_id, _, _ in publication_support.EXPECTED_FOLDS
        for model_id, feature_block_id in PROBABILITY_DESIGN_MATRIX
    )
    batch = publish_x15_oof_batch(
        shard_manifest_paths=tuple(
            shard.manifest_path for shard in shards
        ),
        authority=authority,
        cohort_mapping_sha256=publication_support.MAPPING_SHA,
        output_root=output_root,
    )
    return output_root, authority, shards, batch


def test_real_v4_publisher_public_loader_and_runner_publish_atomically(
    published_selection_batch,
) -> None:
    output_root, authority, _, batch = published_selection_batch

    _, verified_path, verified_sha256, verified_authority = (
        load_verified_x15_oof_selection_batch(
            batch_manifest_path=batch.manifest_path,
            output_root=output_root,
        )
    )
    first = runner.run_stage_b_v2_selection(
        batch_manifest_path=verified_path,
        batch_manifest_file_sha256=verified_sha256,
        oof_output_root=output_root,
        selection_output_root=output_root / "selection",
    )
    second = runner.run_stage_b_v2_selection(
        batch_manifest_path=verified_path,
        batch_manifest_file_sha256=verified_sha256,
        oof_output_root=output_root,
        selection_output_root=output_root / "selection",
    )

    assert verified_authority == authority
    assert first == second
    assert first.manifest_path.is_file()
    document = json.loads(first.manifest_path.read_bytes())
    assert document["input_oof_batch"]["manifest_file_sha256"] == (
        batch.manifest_sha256
    )
    assert document["authority_metadata_sha256"] == authority.metadata_sha256
    assert document["holdout_reaction_accessed"] is False


@pytest.mark.parametrize(
    ("failure", "error_pattern"),
    [
        ("v3", "exact-55|v4"),
        ("partial", "exact-55|selection-ready"),
        ("wrong_survival", "survival_probability_contract"),
    ],
)
def test_real_runner_rejects_non_v4_selection_batch(
    published_selection_batch,
    failure: str,
    error_pattern: str,
) -> None:
    output_root, authority, shards, batch = published_selection_batch
    document = publication_support._manifest(batch.manifest_path)
    if failure == "partial":
        rejected = publish_x15_oof_batch(
            shard_manifest_paths=(shards[0].manifest_path,),
            authority=authority,
            cohort_mapping_sha256=publication_support.MAPPING_SHA,
            output_root=output_root,
        )
        rejected_path = rejected.manifest_path
    else:
        if failure == "v3":
            document["schema"] = "nfl_x15_oof_batch_index_v3"
            document["builder_version"] = "nfl-x15-oof-publication-v3"
        else:
            drifted_run_config = dict(document["batch_run_config"])
            drifted_run_config["survival_probability_contract"] = (
                "CONDITIONAL_HEAD_PRODUCT_V0"
            )
            document["batch_run_config"] = drifted_run_config
            document["batch_run_config_sha256"] = (
                publication_support._sha(drifted_run_config)
            )
            drifted_shared_config = dict(drifted_run_config)
            for field in (
                "fold_ids",
                "model_ids",
                "feature_block_ids",
                "source_shards",
            ):
                drifted_shared_config.pop(field)
            document["shared_run_config_sha256"] = (
                publication_support._sha(drifted_shared_config)
            )
        rejected_path = publication_support._republish_document(
            document,
            output_root=output_root,
            digest_field="batch_sha256",
            suffix=".batch-index.json",
        )
    rejected_sha256 = (
        "sha256:" + hashlib.sha256(rejected_path.read_bytes()).hexdigest()
    )

    with pytest.raises(
        runner.StageBSelectionRunnerError,
        match=error_pattern,
    ):
        runner.run_stage_b_v2_selection(
            batch_manifest_path=rejected_path,
            batch_manifest_file_sha256=rejected_sha256,
            oof_output_root=output_root,
            selection_output_root=output_root / "rejected-selection",
        )
