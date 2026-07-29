from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prediction_market.models import nfl_fastrmodels as frozen
from prediction_market.research import nfl_stage_a_reference as stage_a
from prediction_market.sports.nfl_reference_value import (
    GameReferenceTables,
    ReferenceModelBindingV1,
)


def _registry_row(
    *,
    allowed_experiments: tuple[str, ...] = ("X-11", "X-13", "X-15"),
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_id=frozen.DATASET_ID,
        catalog_item_ids=("I-018",),
        canonical_url=frozen.ASSET_URL,
        license="MIT",
        license_status="approved",
        allowed_experiments=allowed_experiments,
        manifest_sha256=frozen.ASSET_MANIFEST_SHA256,
        status="registered",
    )


class _FakeBooster:
    def __init__(self) -> None:
        self.loaded: bytes | None = None

    def load_model(self, payload: bytearray) -> None:
        self.loaded = bytes(payload)

    def num_features(self) -> int:
        return frozen.EXPECTED_FEATURE_COUNT

    def num_boosted_rounds(self) -> int:
        return frozen.EXPECTED_BOOSTED_ROUNDS

    def set_param(self, params: dict[str, int]) -> None:
        assert params == {"nthread": 1}

    def inplace_predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.full(len(matrix), 0.5, dtype=np.float32)


def test_x15_loader_accepts_only_current_exact_registry_binding(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    verified = SimpleNamespace(object_bytes=b"verified-model")

    monkeypatch.setattr(
        stage_a,
        "load_dataset_registry",
        lambda _root: (_registry_row(),),
    )
    monkeypatch.setattr(
        stage_a._frozen,
        "_require_runtime_version",
        lambda: calls.append("runtime"),
    )
    monkeypatch.setattr(
        stage_a,
        "read_verified_static_object",
        lambda *_args, **_kwargs: verified,
    )
    monkeypatch.setattr(
        stage_a._frozen,
        "_require_verified_asset_identity",
        lambda value: calls.append(
            "identity" if value is verified else "wrong"
        ),
    )
    monkeypatch.setattr(stage_a._frozen.xgboost, "Booster", _FakeBooster)

    predictor = stage_a.load_x15_official_no_spread_predictor(
        program_root=tmp_path,
        store_root=tmp_path,
        manifest_path=tmp_path / "model.manifest.json",
    )

    assert isinstance(predictor, frozen.OfficialNoSpreadPredictor)
    assert calls == ["runtime", "identity"]

    read_called = False

    def forbidden_read(*_args, **_kwargs):
        nonlocal read_called
        read_called = True
        raise AssertionError("asset must not be read after registry mismatch")

    monkeypatch.setattr(
        stage_a,
        "load_dataset_registry",
        lambda _root: (
            _registry_row(allowed_experiments=("X-11", "X-15")),
        ),
    )
    monkeypatch.setattr(stage_a, "read_verified_static_object", forbidden_read)
    with pytest.raises(stage_a.StageAReferenceError, match="dataset registry"):
        stage_a.load_x15_official_no_spread_predictor(
            program_root=tmp_path,
            store_root=tmp_path,
            manifest_path=tmp_path / "model.manifest.json",
        )
    assert read_called is False


def _calibration_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        {
            "game_id": "game-many",
            "state_event_id": f"many-{index}",
            "reference_status": "SUPPORTED",
            "p_home": 0.9,
            "final_home_win": 1,
            "quarter": 1,
            "game_seconds_remaining": 3500 - index,
        }
        for index in range(100)
    )
    rows.append(
        {
            "game_id": "game-one",
            "state_event_id": "one",
            "reference_status": "SUPPORTED",
            "p_home": 0.9,
            "final_home_win": 0,
            "quarter": 4,
            "game_seconds_remaining": 100,
        }
    )
    rows.append(
        {
            "game_id": "game-tie",
            "state_event_id": "tie",
            "reference_status": "SUPPORTED",
            "p_home": 0.1,
            "final_home_win": pd.NA,
            "quarter": 2,
            "game_seconds_remaining": 2000,
        }
    )
    return pd.DataFrame(rows)


def test_calibration_uses_equal_total_game_weight_and_fixed_cluster_bootstrap() -> None:
    first = stage_a.calculate_stage_a_calibration(
        _calibration_rows(),
        bootstrap_samples=40,
        bootstrap_seed=20260729,
    )
    second = stage_a.calculate_stage_a_calibration(
        _calibration_rows(),
        bootstrap_samples=40,
        bootstrap_seed=20260729,
    )

    summary = first.summary.iloc[0]
    # Equal-game Brier: mean((.9-1)^2, (.9-0)^2), not row-micro Brier.
    assert summary["brier"] == pytest.approx(0.41)
    assert summary["evaluated_games"] == 2
    assert summary["evaluated_states"] == 101
    assert summary["excluded_tie_states"] == 1
    assert first.reliability["total_game_weight"].sum() == pytest.approx(2.0)
    assert set(first.breakdowns["dimension"]) == {
        "quarter",
        "game_time",
        "probability",
    }
    pd.testing.assert_frame_equal(first.bootstrap, second.bootstrap)
    assert set(first.bootstrap["metric"]) == {"brier", "log_loss"}
    assert set(first.bootstrap["requested_samples"]) == {40}
    assert set(first.bootstrap["valid_samples"]) == {40}


def test_calibration_rejects_duplicate_state_grain_or_non_supported_values() -> None:
    duplicate = pd.concat(
        [_calibration_rows(), _calibration_rows().iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(stage_a.StageAReferenceError, match="duplicate state"):
        stage_a.calculate_stage_a_calibration(
            duplicate,
            bootstrap_samples=20,
        )

    invalid = _calibration_rows()
    invalid.loc[0, "p_home"] = 1.5
    with pytest.raises(stage_a.StageAReferenceError, match="probabilities"):
        stage_a.calculate_stage_a_calibration(
            invalid,
            bootstrap_samples=20,
        )


def test_workers_must_remain_one() -> None:
    with pytest.raises(stage_a.StageAReferenceError, match="workers must equal 1"):
        stage_a.publish_stage_a_reference_batch(
            project_root="/does/not/matter",
            workers=2,
        )


def _tiny_game_tables() -> GameReferenceTables:
    state_predictions = pd.DataFrame(
        [
            {
                "schema_version": "ReferenceStatePredictionV1",
                "experiment_id": "X-15",
                "claim_boundary": "RETROSPECTIVE_DIAGNOSTIC",
                "game_id": "2025_01_AWY_HME",
                "state_event_id": "event-1",
                "state_raw_play_id": "1",
                "state_atomic_information_episode_id": "episode-1",
                "order_sequence": 1,
                "state_known_at": "2025-09-07T17:00:00Z",
                "quarter": 1,
                "game_seconds_remaining": 3500,
                "home_team": "HME",
                "away_team": "AWY",
                "possession_team": "HME",
                "p_home": 0.6,
                "reference_status": "SUPPORTED",
                "exclusion_reasons": "[]",
                "state_input_sha256": "sha256:" + "d" * 64,
                "model_id": frozen.MODEL_ID,
                "model_version": frozen.ARCHIVE_TAG_COMMIT,
                "model_sha256": frozen.ASSET_SHA256,
                "model_manifest_sha256": frozen.ASSET_MANIFEST_SHA256,
                "pbp_source_sha256": "sha256:" + "c" * 64,
                "final_home_win": 1,
                "final_home_score": 24,
                "final_away_score": 17,
                "evaluation_eligible": True,
                "equal_game_weight": 1.0,
            }
        ]
    )
    feature_provenance = pd.DataFrame(
        [
            {
                "schema_version": "ReferenceFeatureProvenanceV1",
                "experiment_id": "X-15",
                "game_id": "2025_01_AWY_HME",
                "state_event_id": "event-1",
                "state_raw_play_id": "1",
                "state_known_at": "2025-09-07T17:00:00Z",
                "feature_name": "down",
                "feature_value": 1.0,
                "feature_known_at": "2025-09-07T17:00:00Z",
                "source_event_id": "event-1",
                "source_raw_play_id": "1",
                "source_hash": "sha256:" + "c" * 64,
                "pit_status": "PIT_VERIFIED",
            }
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "schema_version": "ReferenceValueObservationV1",
                "experiment_id": "X-15",
                "claim_boundary": "RETROSPECTIVE_DIAGNOSTIC",
                "game_id": "2025_01_AWY_HME",
                "event_id": "event-1",
                "raw_play_id": "1",
                "atomic_information_episode_id": "episode-1",
                "source_interval_start": "2025-09-07T17:00:00Z",
                "source_interval_end": "2025-09-07T17:00:01Z",
                "pre_state_event_id": "event-1",
                "pre_state_raw_play_id": "1",
                "pre_state_known_at": "2025-09-07T17:00:00Z",
                "post_state_event_id": None,
                "post_state_raw_play_id": None,
                "post_state_known_at": None,
                "p_before_home": 0.6,
                "p_after_home": None,
                "reference_delta_home": None,
                "bridge_p_after_home": None,
                "bridge_delta_home": None,
                "intervening_episode_ids": "[]",
                "reference_status": "MISSING_POST_STATE",
                "exclusion_reasons": '["NO_LATER_VALID_CANONICAL_PRE_STATE"]',
                "pre_state_input_sha256": "sha256:" + "d" * 64,
                "post_state_input_sha256": None,
                "model_id": frozen.MODEL_ID,
                "model_sha256": frozen.ASSET_SHA256,
                "model_manifest_sha256": frozen.ASSET_MANIFEST_SHA256,
                "pbp_source_sha256": "sha256:" + "c" * 64,
            }
        ]
    )
    return GameReferenceTables(
        game_id="2025_01_AWY_HME",
        second_half_receiver="HME",
        opening_kickoff_event_id="kickoff",
        opening_kickoff_known_at="2025-09-07T16:59:00Z",
        state_predictions=state_predictions,
        feature_provenance=feature_provenance,
        reference_observations=observations,
    )


def _model_binding() -> ReferenceModelBindingV1:
    return ReferenceModelBindingV1(
        model_id=frozen.MODEL_ID,
        model_version=frozen.ARCHIVE_TAG_COMMIT,
        model_sha256=frozen.ASSET_SHA256,
        manifest_sha256=frozen.ASSET_MANIFEST_SHA256,
        feature_spec_commit=frozen.FEATURE_SPEC_COMMIT,
    )


def test_per_game_publication_is_content_addressed_and_x15_only(tmp_path) -> None:
    input_binding = {
        "experiment_id": "X-13",
        "batch_index_sha256": "sha256:" + "1" * 64,
        "batch_sha256": "sha256:" + "2" * 64,
        "semantic_batch_sha256": "sha256:" + "3" * 64,
        "game_manifest_sha256": "sha256:" + "4" * 64,
        "game_bundle_sha256": "sha256:" + "5" * 64,
        "canonical_events_object_sha256": "sha256:" + "6" * 64,
        "canonical_events_semantic_sha256": "sha256:" + "7" * 64,
    }
    first = stage_a._publish_game_reference_bundle(
        output_root=tmp_path,
        tables=_tiny_game_tables(),
        x13_input_binding=input_binding,
        model=_model_binding(),
        builder_code_sha256="sha256:" + "8" * 64,
    )
    second = stage_a._publish_game_reference_bundle(
        output_root=tmp_path,
        tables=_tiny_game_tables(),
        x13_input_binding=input_binding,
        model=_model_binding(),
        builder_code_sha256="sha256:" + "8" * 64,
    )

    assert first.manifest_path == second.manifest_path
    assert first.manifest_sha256 == second.manifest_sha256
    payload = stage_a._read_json(first.manifest_path, label="test manifest")
    assert payload["schema"] == "nfl_x15_stage_a_single_game_manifest_v1"
    assert payload["experiment_id"] == "X-15"
    assert payload["x13_input"]["experiment_id"] == "X-13"
    assert payload["market_data_read"] is False
    assert payload["holdout_reaction_accessed"] is False
    assert [row["name"] for row in payload["tables"]] == [
        "state_predictions",
        "feature_provenance",
        "reference_observations",
    ]

    first.table_paths[0].write_bytes(b"corrupt")
    with pytest.raises(stage_a.StageAReferenceError, match="collision"):
        stage_a._publish_game_reference_bundle(
            output_root=tmp_path,
            tables=_tiny_game_tables(),
            x13_input_binding=input_binding,
            model=_model_binding(),
            builder_code_sha256="sha256:" + "8" * 64,
        )


def test_batch_publication_binds_games_calibration_and_closed_data_scope(
    tmp_path,
) -> None:
    input_binding = {
        "experiment_id": "X-13",
        "batch_index_sha256": "sha256:" + "1" * 64,
        "batch_sha256": "sha256:" + "2" * 64,
        "semantic_batch_sha256": "sha256:" + "3" * 64,
        "game_manifest_sha256": "sha256:" + "4" * 64,
        "game_bundle_sha256": "sha256:" + "5" * 64,
        "canonical_events_object_sha256": "sha256:" + "6" * 64,
        "canonical_events_semantic_sha256": "sha256:" + "7" * 64,
    }
    game = stage_a._publish_game_reference_bundle(
        output_root=tmp_path,
        tables=_tiny_game_tables(),
        x13_input_binding=input_binding,
        model=_model_binding(),
        builder_code_sha256="sha256:" + "8" * 64,
    )
    calibration = stage_a.calculate_stage_a_calibration(
        _calibration_rows(),
        bootstrap_samples=20,
    )
    x13 = stage_a._VerifiedX13Batch(
        root=tmp_path,
        index_path=tmp_path / "x13.json",
        index_sha256=input_binding["batch_index_sha256"],
        batch_sha256=input_binding["batch_sha256"],
        semantic_batch_sha256=input_binding["semantic_batch_sha256"],
        sources={"pbp": {"dataset_id": "DS-NFLVERSE"}},
        games=(),
    )

    first = stage_a._publish_stage_a_batch_index(
        output_root=tmp_path,
        x13_batch=x13,
        games=(game,),
        calibration=calibration,
        model=_model_binding(),
        builder_code_sha256="sha256:" + "8" * 64,
        bootstrap_samples=20,
        bootstrap_seed=stage_a.BOOTSTRAP_SEED,
        state_status_counts={"SUPPORTED": 1},
        observation_status_counts={"MISSING_POST_STATE": 1},
    )
    second = stage_a._publish_stage_a_batch_index(
        output_root=tmp_path,
        x13_batch=x13,
        games=(game,),
        calibration=calibration,
        model=_model_binding(),
        builder_code_sha256="sha256:" + "8" * 64,
        bootstrap_samples=20,
        bootstrap_seed=stage_a.BOOTSTRAP_SEED,
        state_status_counts={"SUPPORTED": 1},
        observation_status_counts={"MISSING_POST_STATE": 1},
    )

    assert first.index_path == second.index_path
    assert first.index_sha256 == second.index_sha256
    assert len(first.table_paths) == 4
    payload = stage_a._read_json(first.index_path, label="batch index")
    assert payload["schema"] == "nfl_x15_stage_a_batch_index_v1"
    assert payload["experiment_id"] == "X-15"
    assert payload["market_data_read"] is False
    assert payload["holdout_reaction_accessed"] is False
    assert payload["x13_input"]["experiment_id"] == "X-13"
    assert [table["name"] for table in payload["calibration_tables"]] == [
        "calibration_summary",
        "calibration_reliability",
        "calibration_breakdowns",
        "calibration_bootstrap",
    ]


def test_x13_batch_identity_fails_closed_before_any_game_read(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "bad.batch-index.json"
    path.write_text(
        '{"schema":"nfl_x13_exact153_fact_batch_index_v1",'
        '"experiment_id":"X-16","game_count":153}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        stage_a,
        "EXPECTED_X13_BATCH_INDEX_SHA256",
        stage_a._sha256_file(path),
    )

    with pytest.raises(stage_a.StageAReferenceError, match="X-13 batch contract"):
        stage_a._verify_x13_fact_batch(
            project_root=tmp_path,
            batch_index_path=path,
        )


def test_final_outcome_and_equal_game_weight_use_canonical_chronology() -> None:
    canonical = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AWY_HME",
                "order_sequence": 20,
                "raw_play_id": "20",
                "post_home_score": 24,
                "post_away_score": 17,
                # A similarly named retrospective field is not authoritative.
                "final_home_score": 999,
                "final_away_score": 0,
            },
            {
                "game_id": "2025_01_AWY_HME",
                "order_sequence": 10,
                "raw_play_id": "10",
                "post_home_score": 0,
                "post_away_score": 0,
                "final_home_score": 999,
                "final_away_score": 0,
            },
        ]
    )
    states = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AWY_HME",
                "state_event_id": "supported-1",
                "reference_status": "SUPPORTED",
            },
            {
                "game_id": "2025_01_AWY_HME",
                "state_event_id": "excluded",
                "reference_status": "MISSING_PRE_STATE",
            },
            {
                "game_id": "2025_01_AWY_HME",
                "state_event_id": "supported-2",
                "reference_status": "SUPPORTED",
            },
        ]
    )

    enriched = stage_a._attach_final_outcome(states, canonical)

    assert set(enriched["final_home_score"]) == {24}
    assert set(enriched["final_away_score"]) == {17}
    assert set(enriched["final_home_win"]) == {1}
    assert enriched["evaluation_eligible"].tolist() == [True, False, True]
    assert enriched["equal_game_weight"].tolist() == pytest.approx(
        [0.5, 0.0, 0.5]
    )


def test_final_outcome_excludes_ties_and_rejects_non_integral_scores() -> None:
    states = pd.DataFrame(
        [
            {
                "game_id": "2025_04_GB_DAL",
                "state_event_id": "state-1",
                "reference_status": "SUPPORTED",
            }
        ]
    )
    tie = pd.DataFrame(
        [
            {
                "game_id": "2025_04_GB_DAL",
                "order_sequence": 1,
                "raw_play_id": "1",
                "post_home_score": 40,
                "post_away_score": 40,
            }
        ]
    )
    enriched = stage_a._attach_final_outcome(states, tie)
    assert enriched["final_home_win"].isna().all()
    assert enriched["evaluation_eligible"].eq(False).all()
    assert enriched["equal_game_weight"].eq(0.0).all()

    invalid = tie.astype({"post_home_score": "float64"})
    invalid.loc[0, "post_home_score"] = 40.5
    with pytest.raises(stage_a.StageAReferenceError, match="final score"):
        stage_a._attach_final_outcome(states, invalid)
