from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from prediction_market.sports.nfl_reference_value import (
    ReferenceModelProvenanceV1,
    ReferenceValueBuildError,
    build_reference_market_diagnostics,
    build_reference_value_bundle,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _row(
    *,
    game_id: str = "2025_01_AAA_BBB",
    play_id: str,
    episode_id: str,
    order_sequence: int,
    focal_team: str,
    vegas_wp_focal: float,
    quarter: int = 1,
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": game_id,
        "play_id": play_id,
        "episode_id": episode_id,
        "order_sequence": order_sequence,
        "home_team": "BBB",
        "away_team": "AAA",
        "focal_team": focal_team,
        "quarter": quarter,
        "vegas_wp_focal": vegas_wp_focal,
        "event_eligible": True,
        "episode_nullified": False,
        "episode_audit_only": False,
        "event_flags": (),
    }
    row.update(overrides)
    return row


def _precomputed_provenance() -> ReferenceModelProvenanceV1:
    return ReferenceModelProvenanceV1.precomputed(
        model_id="nflfastr-vegas-wp",
        model_version="fastrmodels-pinned-test",
        declared_output_model_sha256=_sha("a"),
        field_name="vegas_wp_focal",
        source_class="open_model",
        pit_status="PIT_DIAGNOSTIC",
        license_status="RESEARCH_ALLOWED",
        calibration_status="UPSTREAM_DIAGNOSTIC",
        formal_registry_gate_passed=True,
    )


def test_builds_episode_boundary_reference_with_home_and_focal_orientation() -> None:
    rows = [
        _row(
            play_id="10",
            episode_id="td-finalized",
            order_sequence=10,
            focal_team="AAA",
            vegas_wp_focal=0.40,
            event_flags=("touchdown",),
        ),
        _row(
            play_id="11",
            episode_id="td-finalized",
            order_sequence=11,
            focal_team="AAA",
            vegas_wp_focal=0.75,
            event_flags=("two_point_attempt",),
        ),
        _row(
            play_id="12",
            episode_id="td-finalized",
            order_sequence=12,
            focal_team="AAA",
            vegas_wp_focal=0.78,
            event_flags=("penalty", "review"),
        ),
        _row(
            play_id="20",
            episode_id="next-live-episode",
            order_sequence=20,
            focal_team="BBB",
            vegas_wp_focal=0.20,
        ),
        _row(
            play_id="30",
            episode_id="following-live-episode",
            order_sequence=30,
            focal_team="AAA",
            vegas_wp_focal=0.70,
        ),
    ]

    bundle = build_reference_value_bundle(
        rows,
        model=_precomputed_provenance(),
        input_source_hashes={"sports_feature_view": _sha("b")},
    )

    td = next(
        row for row in bundle.observations if row.episode_id == "td-finalized"
    )
    assert td.episode_play_ids == ("10", "11", "12")
    assert td.pre_probability_home == pytest.approx(0.60)
    assert td.post_probability_home == pytest.approx(0.20)
    assert td.reference_delta_home == pytest.approx(-0.40)
    assert td.pre_probability_focal == pytest.approx(0.40)
    assert td.post_probability_focal == pytest.approx(0.80)
    assert td.reference_delta_focal == pytest.approx(0.40)
    assert td.model_input_sha256.startswith("sha256:")
    assert td.source_hashes == (_sha("b"),)
    assert td.support_status == "SUPPORTED"
    assert td.formal_comparison_eligible is True
    assert bundle.summary.formal_comparison_eligible == 2
    assert bundle.observations_sha256.startswith("sha256:")


def test_rejects_no_play_and_reversal_and_keeps_ot_as_unproven() -> None:
    rows = [
        _row(
            play_id="1",
            episode_id="no-play",
            order_sequence=1,
            focal_team="AAA",
            vegas_wp_focal=0.50,
            play_type="no_play",
        ),
        _row(
            play_id="2",
            episode_id="reversal",
            order_sequence=2,
            focal_team="BBB",
            vegas_wp_focal=0.55,
            review_result="reversed",
        ),
        _row(
            play_id="3",
            episode_id="ot-play",
            order_sequence=3,
            focal_team="AAA",
            vegas_wp_focal=0.60,
            quarter=5,
        ),
        _row(
            play_id="4",
            episode_id="ot-next",
            order_sequence=4,
            focal_team="BBB",
            vegas_wp_focal=0.30,
            quarter=5,
        ),
        _row(
            play_id="5",
            episode_id="regulation-next",
            order_sequence=5,
            focal_team="AAA",
            vegas_wp_focal=0.65,
            quarter=4,
        ),
    ]

    bundle = build_reference_value_bundle(
        rows,
        model=_precomputed_provenance(),
        input_source_hashes={"sports_feature_view": _sha("c")},
    )
    by_episode = {row.episode_id: row for row in bundle.observations}

    assert by_episode["no-play"].support_status == "EPISODE_INELIGIBLE"
    assert by_episode["no-play"].reference_delta_home is None
    assert by_episode["reversal"].support_status == "EPISODE_INELIGIBLE"
    assert by_episode["reversal"].reference_delta_home is None
    assert (
        by_episode["ot-play"].support_status
        == "MODEL_SUPPORT_UNPROVEN"
    )
    assert by_episode["ot-play"].formal_comparison_eligible is False
    assert by_episode["ot-play"].reference_delta_home is not None
    assert bundle.summary.support_status_counts == (
        ("EPISODE_INELIGIBLE", 2),
        ("MODEL_SUPPORT_UNPROVEN", 2),
        ("MISSING_POST_REFERENCE", 1),
    )


def test_precomputed_model_provenance_is_explicit_and_model_bytes_are_verified() -> None:
    provenance = _precomputed_provenance()

    assert provenance.execution_mode == "PRECOMPUTED_FIELD"
    assert provenance.precomputed_field == "vegas_wp_focal"
    assert provenance.model_bytes_verified is False
    assert provenance.model_sha256 == _sha("a")
    assert provenance.output_model_bytes_status == (
        "OUTPUT_MODEL_BYTES_UNVERIFIED"
    )
    assert provenance.formal_registry_gate_passed is True

    model_bytes = b"pinned-model-bytes"
    digest = "sha256:" + hashlib.sha256(model_bytes).hexdigest()
    verified = ReferenceModelProvenanceV1.from_model_bytes(
        model_id="nflfastr-vegas-wp",
        model_version="fastrmodels-pinned-test",
        model_sha256=digest,
        model_bytes=model_bytes,
        source_class="open_model",
        pit_status="PIT_DIAGNOSTIC",
        license_status="RESEARCH_ALLOWED",
        calibration_status="UPSTREAM_DIAGNOSTIC",
        formal_registry_gate_passed=True,
    )
    assert verified.model_bytes_verified is True
    assert verified.execution_mode == "MODEL_BYTES_VERIFIED"
    assert verified.output_model_bytes_status == "MODEL_BYTES_VERIFIED"
    assert verified.formal_registry_gate_passed is True
    verified_bundle = build_reference_value_bundle(
        [
            _row(
                play_id="1",
                episode_id="one",
                order_sequence=1,
                focal_team="BBB",
                vegas_wp_focal=0.40,
            ),
            _row(
                play_id="2",
                episode_id="two",
                order_sequence=2,
                focal_team="AAA",
                vegas_wp_focal=0.50,
            ),
        ],
        model=verified,
        input_source_hashes={"sports_feature_view": _sha("b")},
    )
    assert verified_bundle.observations[0].model_bytes_verified is True
    assert verified_bundle.observations[0].probability_field == "vegas_wp_focal"

    with pytest.raises(ReferenceValueBuildError, match="model bytes SHA-256"):
        ReferenceModelProvenanceV1.from_model_bytes(
            model_id="nflfastr-vegas-wp",
            model_version="fastrmodels-pinned-test",
            model_sha256=_sha("f"),
            model_bytes=model_bytes,
            source_class="open_model",
            pit_status="PIT_DIAGNOSTIC",
            license_status="RESEARCH_ALLOWED",
            calibration_status="UPSTREAM_DIAGNOSTIC",
            formal_registry_gate_passed=True,
        )


def test_reference_gap_and_fraction_require_material_reference_and_support() -> None:
    rows = [
        _row(
            play_id="1",
            episode_id="material",
            order_sequence=1,
            focal_team="BBB",
            vegas_wp_focal=0.40,
        ),
        _row(
            play_id="2",
            episode_id="small",
            order_sequence=2,
            focal_team="AAA",
            vegas_wp_focal=0.45,
        ),
        _row(
            play_id="3",
            episode_id="small-next",
            order_sequence=3,
            focal_team="BBB",
            vegas_wp_focal=0.56,
            quarter=4,
        ),
        _row(
            play_id="4",
            episode_id="ot",
            order_sequence=4,
            focal_team="BBB",
            vegas_wp_focal=0.44,
            quarter=5,
        ),
        _row(
            play_id="5",
            episode_id="terminal",
            order_sequence=5,
            focal_team="AAA",
            vegas_wp_focal=0.50,
            quarter=5,
        ),
    ]
    bundle = build_reference_value_bundle(
        rows,
        model=_precomputed_provenance(),
        input_source_hashes={"sports_feature_view": _sha("d")},
    )
    diagnostics = build_reference_market_diagnostics(
        bundle,
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "episode_id": "material",
                "logical_market_id": "winner",
                "venue": "polymarket",
                "horizon_seconds": 10,
                "target_team": "BBB",
                "market_delta": 0.10,
                "source_hashes": [_sha("e")],
            },
            {
                "game_id": "2025_01_AAA_BBB",
                "episode_id": "small",
                "logical_market_id": "winner",
                "venue": "kalshi",
                "horizon_seconds": 10,
                "target_team": "AAA",
                "market_delta": 0.02,
                "source_hashes": [_sha("f")],
            },
            {
                "game_id": "2025_01_AAA_BBB",
                "episode_id": "ot",
                "logical_market_id": "winner",
                "venue": "polymarket",
                "horizon_seconds": 10,
                "target_team": "BBB",
                "market_delta": 0.03,
                "source_hashes": [_sha("e")],
            },
        ],
        materiality_threshold=0.05,
    )
    by_episode = {row.episode_id: row for row in diagnostics.rows}

    assert by_episode["material"].reference_delta_target == pytest.approx(0.15)
    assert by_episode["material"].reference_gap == pytest.approx(-0.05)
    assert by_episode["material"].completion_fraction == pytest.approx(2 / 3)
    assert by_episode["material"].comparison_status == "ELIGIBLE"
    assert by_episode["small"].reference_delta_target == pytest.approx(-0.01)
    assert by_episode["small"].reference_gap == pytest.approx(0.03)
    assert by_episode["small"].completion_fraction is None
    assert by_episode["small"].comparison_status == "BELOW_MATERIALITY"
    assert by_episode["ot"].reference_gap is None
    assert by_episode["ot"].comparison_status == "REFERENCE_NOT_SUPPORTED"
    assert diagnostics.rows_sha256.startswith("sha256:")
    small_next = next(
        row
        for row in bundle.observations
        if row.episode_id == "small-next"
    )
    assert small_next.support_status == "MODEL_SUPPORT_UNPROVEN"
    assert small_next.formal_comparison_eligible is False


def test_precomputed_formal_eligibility_requires_explicit_registry_gate() -> None:
    provenance = ReferenceModelProvenanceV1.precomputed(
        model_id="nflfastr-vegas-wp-output",
        model_version="published-field-provenance",
        declared_output_model_sha256=_sha("9"),
        field_name="vegas_wp_focal",
        source_class="open_model",
        pit_status="PIT_DIAGNOSTIC",
        license_status="RESEARCH_ALLOWED",
        calibration_status="UPSTREAM_DIAGNOSTIC",
        formal_registry_gate_passed=False,
    )
    bundle = build_reference_value_bundle(
        [
            _row(
                play_id="1",
                episode_id="one",
                order_sequence=1,
                focal_team="BBB",
                vegas_wp_focal=0.40,
            ),
            _row(
                play_id="2",
                episode_id="two",
                order_sequence=2,
                focal_team="AAA",
                vegas_wp_focal=0.50,
            ),
        ],
        model=provenance,
        input_source_hashes={"sports_feature_view": _sha("b")},
    )
    observation = bundle.observations[0]
    assert observation.support_status == "SUPPORTED"
    assert observation.output_model_bytes_status == (
        "OUTPUT_MODEL_BYTES_UNVERIFIED"
    )
    assert observation.formal_registry_gate_passed is False
    assert observation.formal_comparison_eligible is False

    diagnostic = build_reference_market_diagnostics(
        bundle,
        [
            {
                "game_id": observation.game_id,
                "episode_id": observation.episode_id,
                "logical_market_id": "winner",
                "venue": "polymarket",
                "horizon_seconds": 10,
                "target_team": "BBB",
                "market_delta": 0.05,
                "source_hashes": [_sha("e")],
            }
        ],
        materiality_threshold=0.01,
    ).rows[0]
    assert diagnostic.reference_gap == pytest.approx(-0.05)
    assert diagnostic.completion_fraction is None
    assert diagnostic.comparison_status == "REFERENCE_NOT_SUPPORTED"


def test_observation_contract_rejects_inconsistent_hash_or_support() -> None:
    bundle = build_reference_value_bundle(
        [
            _row(
                play_id="1",
                episode_id="one",
                order_sequence=1,
                focal_team="BBB",
                vegas_wp_focal=0.40,
            ),
            _row(
                play_id="2",
                episode_id="two",
                order_sequence=2,
                focal_team="AAA",
                vegas_wp_focal=0.50,
            ),
        ],
        model=_precomputed_provenance(),
        input_source_hashes={"sports_feature_view": _sha("a")},
    )
    observation = bundle.observations[0]

    with pytest.raises(ReferenceValueBuildError, match="observation_sha256"):
        replace(observation, observation_sha256=_sha("f"))
    with pytest.raises(
        ReferenceValueBuildError, match="formal comparison eligibility"
    ):
        replace(
            observation,
            support_status="MODEL_SUPPORT_UNPROVEN",
            formal_comparison_eligible=True,
        )


def test_duplicate_play_grain_fails_closed() -> None:
    rows = [
        _row(
            play_id="1",
            episode_id="one",
            order_sequence=1,
            focal_team="BBB",
            vegas_wp_focal=0.50,
        ),
        _row(
            play_id="1",
            episode_id="one",
            order_sequence=1,
            focal_team="BBB",
            vegas_wp_focal=0.50,
        ),
    ]

    with pytest.raises(ReferenceValueBuildError, match="duplicate game/play grain"):
        build_reference_value_bundle(
            rows,
            model=_precomputed_provenance(),
            input_source_hashes={"sports_feature_view": _sha("a")},
        )
