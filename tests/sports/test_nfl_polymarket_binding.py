from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from prediction_market.contracts import MarketMetadataSnapshotV0


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "artifacts/game-state/nfl/polymarket_v1_game_binding_audit_v0.json"
)
EXPECTED_GAME_BY_MARKET = {
    "248292": "2022_17_MIA_NE",
    "248293": "2022_17_MIN_GB",
    "248294": "2022_17_PIT_BAL",
}
EXPECTED_TIME_DELTAS = {
    "248292": -145,
    "248293": -43_231,
    "248294": -169,
}


def test_nfl_polymarket_binding_module_exists() -> None:
    assert (
        importlib.util.find_spec(
            "prediction_market.sports.nfl_polymarket_binding"
        )
        is not None
    )


def _module():
    from prediction_market.sports import nfl_polymarket_binding

    return nfl_polymarket_binding


def _gamma_payload(market_id: str) -> dict[str, object]:
    module = _module()
    spec = module.FROZEN_BINDING_BY_MARKET_ID[market_id]
    return json.loads((PROJECT_ROOT / "var/raw" / spec.gamma_object_path).read_bytes())


def _nfl_rows(game_id: str) -> list[dict[str, object]]:
    module = _module()
    payload_path = PROJECT_ROOT / "var/raw" / module.NFLVERSE_OBJECT_PATH
    table = pq.read_table(
        payload_path,
        columns=[
            "game_id",
            "season",
            "home_team",
            "away_team",
            "game_date",
            "season_type",
            "week",
            "start_time",
        ],
        filters=[("game_id", "=", game_id)],
    )
    return table.to_pylist()


def _all_nfl_rows() -> list[dict[str, object]]:
    module = _module()
    payload_path = PROJECT_ROOT / "var/raw" / module.NFLVERSE_OBJECT_PATH
    return pq.read_table(
        payload_path,
        columns=[
            "game_id",
            "season",
            "home_team",
            "away_team",
            "game_date",
            "season_type",
            "week",
            "start_time",
        ],
    ).to_pylist()


def _gamma_identities() -> dict[str, object]:
    module = _module()
    return {
        spec.condition_id: module.parse_gamma_market_identity(
            _gamma_payload(spec.gamma_market_id), spec
        )
        for spec in module.FROZEN_BINDINGS
    }


def _extract_payload() -> bytes:
    module = _module()
    return (
        PROJECT_ROOT / "var/raw" / module.POLYMARKET_V1_EXTRACT_OBJECT_PATH
    ).read_bytes()


def test_production_specs_do_not_preseed_nflverse_binding_answers() -> None:
    module = _module()

    for spec in module.FROZEN_BINDINGS:
        for forbidden in (
            "native_nflverse_game_id",
            "away_team",
            "home_team",
            "game_date",
            "native_start_time",
            "cancelled",
        ):
            assert not hasattr(spec, forbidden)


def test_gamma_text_parses_team_aliases_and_game_date() -> None:
    module = _module()
    spec = module.FROZEN_BINDING_BY_MARKET_ID["248293"]

    identity = module.parse_gamma_market_identity(_gamma_payload("248293"), spec)

    assert identity.outcome_aliases == ("Vikings", "Packers")
    assert identity.full_team_names == (
        "Minnesota Vikings",
        "Green Bay Packers",
    )
    assert identity.nfl_abbreviations == ("MIN", "GB")
    assert identity.game_date == "2023-01-01"


def test_vikings_cannot_be_bound_to_miami_game_by_caller_choice() -> None:
    module = _module()
    spec = module.FROZEN_BINDING_BY_MARKET_ID["248293"]
    identity = module.parse_gamma_market_identity(_gamma_payload("248293"), spec)

    candidate = module.find_unique_nflverse_candidate(
        _nfl_rows("2022_17_MIA_NE"), identity
    )

    assert candidate is None


def test_completed_binding_boundary_rejects_foreign_candidate() -> None:
    module = _module()
    viking_source = module.FROZEN_BINDING_BY_MARKET_ID["248293"]
    miami_source = module.FROZEN_BINDING_BY_MARKET_ID["248292"]
    viking_identity = module.parse_gamma_market_identity(
        _gamma_payload("248293"), viking_source
    )
    miami_identity = module.parse_gamma_market_identity(
        _gamma_payload("248292"), miami_source
    )
    miami_candidate = module.find_unique_nflverse_candidate(
        _nfl_rows(EXPECTED_GAME_BY_MARKET["248292"]), miami_identity
    )
    assert miami_candidate is not None

    with pytest.raises(module.NFLPolymarketBindingError):
        module.validate_candidate_identity(viking_identity, miami_candidate)


def test_candidate_native_game_id_must_match_candidate_fields() -> None:
    module = _module()
    source = module.FROZEN_BINDING_BY_MARKET_ID["248293"]
    identity = module.parse_gamma_market_identity(
        _gamma_payload("248293"), source
    )
    candidate = module.find_unique_nflverse_candidate(_all_nfl_rows(), identity)
    assert candidate is not None
    forged = replace(candidate, native_game_id="2022_17_MIA_NE")

    with pytest.raises(module.NFLPolymarketBindingError):
        module.validate_candidate_identity(identity, forged)


def test_unique_team_pair_and_date_candidate_is_discovered_not_preseeded() -> None:
    module = _module()
    spec = module.FROZEN_BINDING_BY_MARKET_ID["248293"]
    identity = module.parse_gamma_market_identity(_gamma_payload("248293"), spec)

    candidate = module.find_unique_nflverse_candidate(_all_nfl_rows(), identity)

    assert candidate is not None
    assert candidate.native_game_id == "2022_17_MIN_GB"
    assert candidate.away_team == "MIN"
    assert candidate.home_team == "GB"
    assert candidate.game_date == "2023-01-01"
    assert candidate.row_count == 166


def test_duplicate_team_pair_and_date_candidates_fail_closed() -> None:
    module = _module()
    spec = module.FROZEN_BINDING_BY_MARKET_ID["248293"]
    identity = module.parse_gamma_market_identity(_gamma_payload("248293"), spec)
    rows = _nfl_rows("2022_17_MIN_GB")
    duplicate = copy.deepcopy(rows)
    for row in duplicate:
        row["game_id"] = "2022_17_MIN_GB_DUPLICATE"

    with pytest.raises(module.NFLPolymarketBindingError):
        module.find_unique_nflverse_candidate([*rows, *duplicate], identity)


def test_unexpected_large_time_delta_fails_closed() -> None:
    module = _module()

    with pytest.raises(module.NFLPolymarketBindingError):
        module.compare_source_starts(
            gamma_game_start_at="2023-01-01T09:25:00Z",
            nflverse_start_time_native="1/1/23, 13:02:25",
        )


def test_real_frozen_binding_audit_is_exact_and_fail_closed() -> None:
    module = _module()

    audit = module.build_nfl_polymarket_binding_audit(PROJECT_ROOT)

    assert audit["status"] == "RETROSPECTIVE_RESEARCH_ONLY"
    assert audit["summary"] == {
        "frozen_condition_count": 4,
        "exact_game_binding_count": 3,
        "unmatched_cancelled_count": 1,
        "canonical_outcome_document_count": 6,
        "model_output_count": 0,
        "matched_as_of_rows": 0,
    }
    assert len(audit["input_objects"]) == 6
    assert {
        row["gamma_market_id"]: row["native_nflverse_game_id"]
        for row in audit["exact_game_bindings"]
    } == EXPECTED_GAME_BY_MARKET
    assert {
        row["gamma_market_id"]: row["source_time_comparison"][
            "gamma_minus_nflverse_seconds"
        ]
        for row in audit["exact_game_bindings"]
    } == EXPECTED_TIME_DELTAS

    viking = next(
        row
        for row in audit["exact_game_bindings"]
        if row["gamma_market_id"] == "248293"
    )
    assert viking["source_time_comparison"]["anomaly_detected"] is True
    assert viking["source_time_comparison"]["anomaly_kind"] == (
        "gamma_game_start_approximately_12_hours_early"
    )
    assert viking["source_time_comparison"]["correction_applied"] is False
    for row in audit["exact_game_bindings"]:
        if row["gamma_market_id"] != "248293":
            assert row["source_time_comparison"]["anomaly_detected"] is False

    cancelled = audit["unmatched_conditions"]
    assert len(cancelled) == 1
    assert cancelled[0]["gamma_market_id"] == "248295"
    assert cancelled[0]["candidate_native_nflverse_game_id"] is None
    assert cancelled[0]["identity"]["nfl_team_abbreviations"] == ["BUF", "CIN"]
    assert cancelled[0]["status"] == "UNMATCHED_CANCELLED"
    assert cancelled[0]["nflverse_row_count"] == 0
    assert cancelled[0]["hard_binding_created"] is False


def test_six_outcome_documents_validate_existing_snapshot_contract() -> None:
    module = _module()

    audit = module.build_nfl_polymarket_binding_audit(PROJECT_ROOT)
    documents = [
        document
        for binding in audit["exact_game_bindings"]
        for document in binding["canonical_outcome_documents"]
    ]

    assert len(documents) == 6
    assert len({document["snapshot_sha256"] for document in documents}) == 6
    for document in documents:
        snapshot = MarketMetadataSnapshotV0.model_validate(document)
        assert snapshot.captured_at > snapshot.game_start_at
        assert snapshot.canonical_refs.game_id.startswith("game_nflverse_2022_17_")
        assert snapshot.native_condition_id.startswith("0x")
        assert snapshot.native_token_id.isdigit()
    by_market = {
        binding["gamma_market_id"]: binding["canonical_outcome_documents"]
        for binding in audit["exact_game_bindings"]
    }
    assert [
        document["native_outcome_id"] for document in by_market["248292"]
    ] == ["Dolphins", "Patriots"]
    assert [
        document["native_outcome_id"] for document in by_market["248293"]
    ] == ["Vikings", "Packers"]
    assert [
        document["native_outcome_id"] for document in by_market["248294"]
    ] == ["Steelers", "Ravens"]


def test_audit_bytes_are_deterministic_and_match_checked_in_artifact() -> None:
    module = _module()

    first = module.canonical_audit_bytes(
        module.build_nfl_polymarket_binding_audit(PROJECT_ROOT)
    )
    second = module.canonical_audit_bytes(
        module.build_nfl_polymarket_binding_audit(PROJECT_ROOT)
    )

    assert first == second
    assert ARTIFACT_PATH.read_bytes() == first


def test_artifact_states_retrospective_evidence_boundary() -> None:
    module = _module()

    audit = module.build_nfl_polymarket_binding_audit(PROJECT_ROOT)
    boundary = audit["evidence_boundary"]

    assert boundary["intended_use"] == "RETROSPECTIVE_RESEARCH_ONLY"
    assert boundary["gamma_license_status"] == "pending"
    assert boundary["metadata_fetched_after_event"] is True
    assert boundary["point_in_time_for_game_state_or_model"] is False
    assert boundary["fills_are_level2"] is False
    assert boundary["local_receive_time_available"] is False
    assert boundary["executable_depth_available"] is False
    assert boundary["venue_rule_snapshot_available"] is False
    assert boundary["model_output_count"] == 0
    assert boundary["matched_as_of_rows"] == 0
    assert boundary["symmetry_status"] == "NOT_VERIFIED"
    assert boundary["alpha_status"] == "NOT_VERIFIED"
    assert boundary["profit_or_return_computed"] is False
    assert boundary["fill_prices_treated_as_two_sided_quotes"] is False


@pytest.mark.parametrize(
    "mutation",
    ["payload_bytes", "manifest_object_hash", "manifest_hash"],
)
def test_frozen_object_hash_tampering_fails_closed(mutation: str) -> None:
    module = _module()
    spec = module.FROZEN_BINDING_BY_MARKET_ID["248292"]
    payload = (PROJECT_ROOT / "var/raw" / spec.gamma_object_path).read_bytes()
    manifest = json.loads(
        (PROJECT_ROOT / "var/raw" / spec.gamma_manifest_path).read_bytes()
    )
    if mutation == "payload_bytes":
        payload += b" "
    elif mutation == "manifest_object_hash":
        manifest["object_sha256"] = "sha256:" + ("0" * 64)
    else:
        manifest["manifest_sha256"] = "sha256:" + ("0" * 64)

    with pytest.raises(module.NFLPolymarketBindingError):
        module.verify_frozen_object(
            program_root=PROJECT_ROOT,
            payload=payload,
            manifest=manifest,
            expected_object_sha256=spec.gamma_object_sha256,
            expected_manifest_sha256=spec.gamma_manifest_sha256,
        )


@pytest.mark.parametrize("field", ["away_team", "home_team", "game_date"])
def test_team_or_date_tampering_removes_unique_candidate(field: str) -> None:
    module = _module()
    spec = module.FROZEN_BINDING_BY_MARKET_ID["248292"]
    identity = module.parse_gamma_market_identity(_gamma_payload("248292"), spec)
    rows = _nfl_rows(EXPECTED_GAME_BY_MARKET["248292"])
    tampered = copy.deepcopy(rows)
    tampered[0][field] = "TAMPERED"

    with pytest.raises(module.NFLPolymarketBindingError):
        module.find_unique_nflverse_candidate(tampered, identity)


@pytest.mark.parametrize("mutation", ["condition", "outcomes", "tokens"])
def test_condition_or_outcome_orientation_tampering_fails_closed(
    mutation: str,
) -> None:
    module = _module()
    spec = module.FROZEN_BINDING_BY_MARKET_ID["248292"]
    payload = _gamma_payload("248292")
    if mutation == "condition":
        payload["conditionId"] = "0x" + ("0" * 64)
    elif mutation == "outcomes":
        payload["outcomes"] = json.dumps(
            list(reversed(json.loads(str(payload["outcomes"]))))
        )
    else:
        payload["clobTokenIds"] = json.dumps(
            list(reversed(json.loads(str(payload["clobTokenIds"]))))
        )

    if mutation in {"condition", "outcomes"}:
        with pytest.raises(module.NFLPolymarketBindingError):
            module.parse_gamma_market_identity(payload, spec)
    else:
        identities = _gamma_identities()
        identities[spec.condition_id] = module.parse_gamma_market_identity(
            payload, spec
        )
        with pytest.raises(module.NFLPolymarketBindingError):
            module.validate_bounded_extract_against_gamma(
                _extract_payload(), identities
            )


def test_extract_winner_must_match_gamma_terminal_price_orientation() -> None:
    module = _module()
    condition_id = module.FROZEN_BINDING_BY_MARKET_ID["248292"].condition_id
    mutated_lines = []
    for line in _extract_payload().splitlines():
        item = json.loads(line)
        if item["row"]["condition_id"] == condition_id:
            item["row"]["winning_outcome_label"] = "Dolphins"
        mutated_lines.append(
            json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    mutated = b"\n".join(mutated_lines) + b"\n"

    with pytest.raises(module.NFLPolymarketBindingError):
        module.validate_bounded_extract_against_gamma(
            mutated, _gamma_identities()
        )


def test_artifact_carries_and_recomputes_canonical_self_hash() -> None:
    module = _module()
    audit = module.build_nfl_polymarket_binding_audit(PROJECT_ROOT)

    assert audit["artifact_sha256"] == (
        module.nfl_polymarket_binding_audit_sha256(audit)
    )
    assert module.load_nfl_polymarket_binding_audit(ARTIFACT_PATH) == audit

    tampered = copy.deepcopy(audit)
    tampered["summary"]["matched_as_of_rows"] = 1
    with pytest.raises(module.NFLPolymarketBindingError):
        module.validate_nfl_polymarket_binding_audit(tampered)


def test_cancelled_game_must_remain_absent() -> None:
    module = _module()
    spec = module.FROZEN_BINDING_BY_MARKET_ID["248295"]
    identity = module.parse_gamma_market_identity(_gamma_payload("248295"), spec)

    assert module.find_unique_nflverse_candidate(_all_nfl_rows(), identity) is None
