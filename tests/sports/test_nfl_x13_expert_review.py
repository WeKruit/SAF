from __future__ import annotations

import json
import hashlib
import copy
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from prediction_market.sports.nfl_x13_expert_review import (
    ExpertReviewRecord,
    ReviewDecision,
    SportsExpertReviewError,
    build_x13_sports_expert_review_payload,
    canonical_x13_discovery_sha256,
    canonical_x13_factor_registry_sha256,
    canonical_payload_sha256,
    freeze_verified_x13_expert_shortlist,
    load_x13_frozen_factor_registry,
    publish_x13_candidate_expert_review_artifact,
    seed_x13_sports_expert_reviews,
    verify_x13_candidate_expert_review_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT_ROOT / "registries/factors/nfl_x13_sports_expert_review_v2.json"
FACTOR_REGISTRY = PROJECT_ROOT / "registries/factors/nfl_factor_registry_v2.json"


def _record(**changes: object) -> ExpertReviewRecord:
    values: dict[str, object] = {
        "factor_id": "NFL.EVENT.SACK",
        "definition_sha256": "sha256:" + "a" * 64,
        "decision": ReviewDecision.ACCEPT,
        "actor": "防守方",
        "beneficiary": "防守方",
        "focal_orientation": "以受益球队为焦点；对手为反向。",
        "overlap_factor_ids": ("NFL.EVENT.PASS",),
        "required_control_fields": (
            "pre_event_probability",
            "game_seconds_remaining",
        ),
        "rationale_zh": "擒杀是最终可核验的原子比赛事件。",
    }
    values.update(changes)
    return ExpertReviewRecord(**values)  # type: ignore[arg-type]


def test_review_record_validates_decision_digest_and_semantic_bindings() -> None:
    record = _record()

    assert record.decision is ReviewDecision.ACCEPT
    assert record.shortlist_eligible is False
    assert record.as_payload()["actor"] == "防守方"

    with pytest.raises(SportsExpertReviewError, match="shortlist_eligible"):
        _record(shortlist_eligible=True)
    with pytest.raises(SportsExpertReviewError, match="definition_sha256"):
        _record(definition_sha256="not-a-digest")
    with pytest.raises(SportsExpertReviewError, match="rationale_zh"):
        _record(rationale_zh="English only")


def test_seeded_decisions_capture_the_current_sports_semantic_calls() -> None:
    records = {record.factor_id: record for record in seed_x13_sports_expert_reviews()}

    assert records["NFL.TURNOVER.INTERCEPTION"].decision is ReviewDecision.ACCEPT
    assert records["NFL.SCORE.PASS_TD"].decision is ReviewDecision.ACCEPT
    assert records["NFL.SEQUENCE.RED_ZONE_EMPTY_DRIVE"].decision is ReviewDecision.REDEFINE
    assert records["NFL.EVENT.FUMBLE_RECOVERY"].decision is ReviewDecision.REDEFINE
    assert records["NFL.COMBO.THIRD_FOURTH_DOWN_FAILURE"].decision is ReviewDecision.REDEFINE
    assert records["NFL.SPECIAL.PUNT_POSITIVE_RETURN"].decision is ReviewDecision.REDEFINE
    assert records["NFL.SCORE.FG_MADE"].decision is ReviewDecision.REDEFINE
    for factor_id in (
        "NFL.SCORE.PAT_MADE",
        "NFL.EVENT.PASS",
        "NFL.EVENT.RUSH",
        "NFL.SPECIAL.PUNT",
        "NFL.EVENT.SUCCESS",
        "NFL.COMBO.DISTANCE_GT_10",
        "NFL.STATE.QUARTER_4",
    ):
        assert records[factor_id].decision is ReviewDecision.REJECT
    assert all(record.shortlist_eligible is False for record in records.values())
    assert all(record.rationale_zh for record in records.values())


def test_review_payload_is_deterministic_and_binds_every_required_field() -> None:
    payload = build_x13_sports_expert_review_payload(
        seed_x13_sports_expert_reviews()
    )

    assert payload["schema_version"] == "NFLX13SportsExpertReviewV2"
    assert payload["shortlist_status"] == "CLOSED_BY_DEFAULT"
    assert payload["claim_boundary"] == "PRELIMINARY_SOURCE_TIME_ONLY"
    assert [item["factor_id"] for item in payload["reviews"]][:2] == [
        "NFL.COMBO.DISTANCE_GT_10",
        "NFL.COMBO.THIRD_FOURTH_DOWN_FAILURE",
    ]
    assert set(payload["reviews"][0]) >= {
        "definition_sha256",
        "actor",
        "beneficiary",
        "focal_orientation",
        "overlap_factor_ids",
        "required_control_fields",
        "rationale_zh",
        "shortlist_eligible",
    }
    assert payload["payload_sha256"] == canonical_payload_sha256(payload)
    assert json.loads(json.dumps(payload, ensure_ascii=False))["reviews"][0][
        "shortlist_eligible"
    ] is False


def test_build_rejects_unfrozen_factor_hash_overlap_and_control_bindings() -> None:
    record = seed_x13_sports_expert_reviews()[0]

    with pytest.raises(SportsExpertReviewError, match="definition_sha256"):
        build_x13_sports_expert_review_payload(
            [replace(record, definition_sha256="sha256:" + "d" * 64)],
        )
    with pytest.raises(SportsExpertReviewError, match="factor_id"):
        build_x13_sports_expert_review_payload(
            [replace(record, factor_id="NFL.EVENT.UNKNOWN")],
        )
    with pytest.raises(SportsExpertReviewError, match="overlap_factor_ids"):
        build_x13_sports_expert_review_payload(
            [replace(record, overlap_factor_ids=("NFL.EVENT.UNKNOWN",))],
        )
    with pytest.raises(SportsExpertReviewError, match="required_control_fields"):
        build_x13_sports_expert_review_payload(
            [replace(record, required_control_fields=("not_an_allowed_control",))],
        )


def test_loader_rejects_bytes_that_do_not_match_the_canonical_registry(
    tmp_path: Path,
) -> None:
    fake = tmp_path / "nfl_factor_registry_v2.json"
    fake.write_bytes(FACTOR_REGISTRY.read_bytes() + b"\n")

    with pytest.raises(SportsExpertReviewError, match="physical hash"):
        load_x13_frozen_factor_registry(fake)


def test_public_builder_does_not_allow_control_allowlist_override() -> None:
    with pytest.raises(TypeError, match="allowed_control_fields"):
        build_x13_sports_expert_review_payload(  # type: ignore[call-arg]
            seed_x13_sports_expert_reviews(),
            allowed_control_fields=frozenset({"not_an_allowed_control"}),
        )


def test_v2_registry_is_content_addressed_superseding_seeded_review_payload() -> None:
    payload = json.loads(REGISTRY.read_text("utf-8"))
    frozen = load_x13_frozen_factor_registry(FACTOR_REGISTRY)

    assert payload == build_x13_sports_expert_review_payload(
        seed_x13_sports_expert_reviews(),
    )
    assert payload["supersedes"] == ["nfl_factor_registry_v2.json"]
    assert payload["payload_sha256"] == canonical_payload_sha256(payload)
    assert frozen["NFL.EVENT.SACK"] == payload["reviews"][5]["definition_sha256"]


def _shortlist_factor_registry() -> list[dict[str, object]]:
    return [
        {
            "factor_id": "NFL.TEST.SACK",
            "version": "v1",
            "sports_meaning": "A test-only structured sack event.",
            "predicate": [{"field": "sack", "operator": "eq", "value": True}],
            "pit_fields": ["sack"],
            "focal_entity": "team",
            "exclusions": [],
            "target": "signed_price_change",
            "support_kind": "named",
            "market_families": ["moneyline"],
            "horizons_seconds": [10],
            "minimum_episodes": 20,
            "minimum_games": 6,
            "minimum_episodes_per_venue": 0,
            "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        }
    ]


def _shortlist_discovery() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "factor_id": "NFL.TEST.SACK",
                "factor_version": "v1",
                "market_family": "moneyline",
                "horizon_seconds": 10,
                "target": "signed_price_change",
                "discovery_status": "AWAITING_EXPERT_REVIEW",
                "point_estimate": 0.01,
                "ci_low": 0.005,
                "ci_high": 0.015,
                "bh_q_value": 0.01,
                "loo_sign_stability_rate": 1.0,
                "maximum_single_game_contribution": 0.1,
                "venue_sign_consistent": True,
                "support_gate_pass": True,
            }
        ]
    )


def _matched_artifact(tmp_path: Path) -> Path:
    payload = b'{"schema":"matched-test-v1","rows":1}'
    artifact_sha256 = "sha256:" + hashlib.sha256(payload).hexdigest()
    hexadecimal = artifact_sha256.removeprefix("sha256:")
    path = (
        tmp_path
        / f"sha256-{hexadecimal}"
        / f"{hexadecimal}.matched.json"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return path


def _candidate_artifact(tmp_path: Path) -> tuple[Path, Path]:
    matched_path = _matched_artifact(tmp_path / "matched")
    review = publish_x13_candidate_expert_review_artifact(
        output_root=tmp_path / "expert-reviews",
        experiment_id="X-13",
        factor_id="NFL.TEST.SACK",
        factor_version="v1",
        market_family="moneyline",
        horizon_seconds=10,
        decision="ACCEPT",
        reviewer="human:sports_expert",
        reviewed_at_utc="2026-07-27T12:00:00Z",
        sports_rationale="The final sack outcome has a coherent football mechanism.",
        anomaly_case_ids=("CASE-001",),
        priority="P1",
        discovery_sha256=canonical_x13_discovery_sha256(_shortlist_discovery()),
        factor_registry_sha256=canonical_x13_factor_registry_sha256(
            _shortlist_factor_registry()
        ),
        matched_sha256="sha256:"
        + hashlib.sha256(matched_path.read_bytes()).hexdigest(),
    )
    return review.artifact_path, matched_path


def test_candidate_review_artifact_rejects_tampering_and_source_mismatch(
    tmp_path: Path,
) -> None:
    artifact_path, _matched_path = _candidate_artifact(tmp_path)
    raw = json.loads(artifact_path.read_text("utf-8"))
    raw["priority"] = "P0"
    artifact_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(SportsExpertReviewError, match="hash"):
        verify_x13_candidate_expert_review_artifact(artifact_path)

    clean_artifact, clean_matched = _candidate_artifact(tmp_path / "clean")
    forged = json.loads(clean_artifact.read_text("utf-8"))
    forged["schema"] = "forged_candidate_review_v1"
    forged_material = {
        key: value for key, value in forged.items() if key != "artifact_sha256"
    }
    forged_sha256 = "sha256:" + hashlib.sha256(
        json.dumps(
            forged_material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    forged["artifact_sha256"] = forged_sha256
    forged_path = (
        tmp_path
        / f"sha256-{forged_sha256.removeprefix('sha256:')}"
        / f"{forged_sha256.removeprefix('sha256:')}.candidate-expert-review.json"
    )
    forged_path.parent.mkdir()
    forged_path.write_text(
        json.dumps(
            forged,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    with pytest.raises(SportsExpertReviewError, match="schema"):
        verify_x13_candidate_expert_review_artifact(forged_path)

    with pytest.raises(SportsExpertReviewError, match="discovery_sha256"):
        freeze_verified_x13_expert_shortlist(
            review_artifact_paths=[clean_artifact],
            discovery_results=_shortlist_discovery().assign(point_estimate=0.02),
            factor_registry=_shortlist_factor_registry(),
            matched_artifact_path=clean_matched,
            output_root=tmp_path / "shortlists",
        )

    mutated_registry = copy.deepcopy(_shortlist_factor_registry())
    mutated_registry[0]["sports_meaning"] = "Altered definition."
    with pytest.raises(SportsExpertReviewError, match="factor_registry_sha256"):
        freeze_verified_x13_expert_shortlist(
            review_artifact_paths=[clean_artifact],
            discovery_results=_shortlist_discovery(),
            factor_registry=mutated_registry,
            matched_artifact_path=clean_matched,
            output_root=tmp_path / "shortlists-registry",
        )


def test_valid_candidate_artifact_passes_actual_shortlist_lock(
    tmp_path: Path,
) -> None:
    artifact_path, matched_path = _candidate_artifact(tmp_path)
    verified = verify_x13_candidate_expert_review_artifact(artifact_path)

    lock = freeze_verified_x13_expert_shortlist(
        review_artifact_paths=[artifact_path],
        discovery_results=_shortlist_discovery(),
        factor_registry=_shortlist_factor_registry(),
        matched_artifact_path=matched_path,
        output_root=tmp_path / "shortlists",
    )

    assert verified.factor_id == "NFL.TEST.SACK"
    assert lock.candidate_keys == ("NFL.TEST.SACK|moneyline|h=10",)
    payload = json.loads(lock.manifest_path.read_text("utf-8"))
    assert payload["human_reviews"] == [
        {
            "anomaly_case_ids": ["CASE-001"],
            "decision": "ACCEPT",
            "factor_id": "NFL.TEST.SACK",
            "factor_version": "v1",
            "horizon_seconds": 10,
            "market_family": "moneyline",
            "priority": "P1",
            "reviewed_at_utc": "2026-07-27T12:00:00Z",
            "reviewer": "human:sports_expert",
            "sports_rationale": "The final sack outcome has a coherent football mechanism.",
        }
    ]
