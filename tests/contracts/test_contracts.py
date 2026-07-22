from __future__ import annotations

import copy
import hashlib
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_ROOT = PROJECT_ROOT / "contracts"
ADR_ROOT = PROJECT_ROOT / "artifacts" / "architecture" / "adr"

sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _contracts():
    try:
        from prediction_market import contracts
    except ModuleNotFoundError:
        pytest.fail("prediction_market.contracts has not been implemented")
    return contracts


def _digest(fill: str = "0") -> str:
    return f"sha256:{fill * 64}"


def _event_id(fill: str = "0") -> str:
    return f"evt_{fill * 64}"


def _canonical_refs() -> dict[str, Any]:
    return {
        "competition_id": "cmp_nba",
        "game_id": "game_nba_2026_001",
        "participant_ids": ["participant_away", "participant_home"],
        "venue_event_id": "venue_event_polymarket_001",
        "market_id": "market_moneyline_001",
        "outcome_id": "outcome_home_win",
        "condition_id": "condition_0xabc",
    }


def _raw_event() -> dict[str, Any]:
    contracts = _contracts()
    event: dict[str, Any] = {
        "envelope_version": "v0",
        "event_type": "raw_observation",
        "payload_schema_version": "v0",
        "source": {
            "system": "pmxt-archive",
            "stream": "polymarket-market-ws",
            "venue": "polymarket",
            "sequence": 17,
            "capture_session_id": "capture_20260722T120000Z",
            "record_ordinal": 23,
        },
        "time": {
            "receive_at": "2026-07-22T12:00:00.123456Z",
            "receive_basis": "upstream_exporter",
            "source_at": "2026-07-22T12:00:00Z",
            "publish_at": None,
            "exchange_at": "2026-07-22T12:00:00.100Z",
        },
        "canonical_refs": _canonical_refs(),
        "native_refs": [
            {
                "namespace": "polymarket.condition",
                "native_id": "0xabc",
            }
        ],
        "lineage": {
            "raw_object_hash": _digest("a"),
            "raw_record_ordinal": 23,
        },
        "experiment_id": "X-01",
        "rule_snapshot_ref": None,
        "quality_flags": ["gap_detected", "out_of_order"],
        "payload": {
            "asset_id": "asset-1",
            "book": {
                "bids": [
                    {"price": {"atoms": "49", "scale": 2}, "size": {"atoms": "10", "scale": 0}}
                ]
            },
        },
    }
    event["payload_sha256"] = contracts.payload_sha256(event["payload"])
    event["event_id"] = contracts.event_id_for(event)
    return event


def _derived_event() -> dict[str, Any]:
    contracts = _contracts()
    event: dict[str, Any] = {
        "envelope_version": "v0",
        "event_type": "model_output",
        "payload_schema_version": "v0",
        "source": {
            "system": "nba-baseline",
            "stream": "state-transition-output",
            "venue": None,
            "sequence": None,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        "time": {
            "receive_at": "2026-07-22T12:00:01Z",
            "receive_basis": "local_recorder",
            "source_at": "2026-07-22T12:00:00Z",
            "publish_at": None,
            "exchange_at": None,
        },
        "canonical_refs": _canonical_refs(),
        "native_refs": [],
        "lineage": {"parent_event_ids": [_event_id("a"), _event_id("b")]},
        "experiment_id": "X-06",
        "rule_snapshot_ref": _digest("c"),
        "quality_flags": ["source_clock_unverified", "out_of_order"],
        "payload": {"state": "home_possession"},
    }
    event["payload_sha256"] = contracts.payload_sha256(event["payload"])
    event["event_id"] = contracts.event_id_for(event)
    return event


def _model_output() -> dict[str, Any]:
    return {
        "contract_version": "v0",
        "model_id": "nba-state-baseline",
        "model_version": "2026-07-22.1",
        "experiment_id": "X-06",
        "run_id": "run_x06_001",
        "game_id": "game_nba_2026_001",
        "state_event_id": _event_id("d"),
        "pit_cutoff_at": "2026-07-22T12:00:00Z",
        "state_space": ["away_possession", "home_possession", "game_over"],
        "horizon": "next_state_transition",
        "probabilities": {
            "away_possession": {"atoms": "25", "scale": 2},
            "home_possession": {"atoms": "50", "scale": 2},
            "game_over": {"atoms": "25", "scale": 2},
        },
        "feature_sha256": _digest("1"),
        "data_sha256": _digest("2"),
        "config_sha256": _digest("3"),
        "quality_flags": [],
    }


def _rule_snapshot() -> dict[str, Any]:
    return {
        "venue": "polymarket",
        "condition_id": "condition_0xabc",
        "fetched_at": "2026-07-22T12:00:00Z",
        "effective_from": "2026-07-22T12:00:00Z",
        "game_start_time": "2026-07-22T23:00:00Z",
        "seconds_delay": {"atoms": "1", "scale": 0},
        "cancel_during_delay": False,
        "start_time_cancel_policy": "cancel_all_with_schedule_change_exception",
        "fees_enabled": True,
        "fee_rate": {"atoms": "5", "scale": 2},
        "fee_exponent": {"atoms": "1", "scale": 0},
        "taker_only": True,
        "maker_fee_rate": {"atoms": "0", "scale": 0},
        "minimum_tick_size": {"atoms": "1", "scale": 2},
        "minimum_order_size": {"atoms": "5", "scale": 0},
        "order_types_supported": ["FAK", "FOK", "GTC", "GTD"],
        "source_document_version": "docs.polymarket.com@2026-07-22",
        "raw_response_hash": _digest("e"),
    }


@pytest.mark.parametrize(
    "atoms",
    ["", "+1", "01", "-01", "-0", " 1", "1 ", "1.0", "1e2", "--1"],
)
def test_fixed_point_rejects_noncanonical_atoms(atoms: str) -> None:
    FixedPointV0 = _contracts().FixedPointV0

    with pytest.raises(ValidationError):
        FixedPointV0(atoms=atoms, scale=0)


@pytest.mark.parametrize("scale", [-1, 19])
def test_fixed_point_rejects_scale_outside_v0_bounds(scale: int) -> None:
    FixedPointV0 = _contracts().FixedPointV0

    with pytest.raises(ValidationError):
        FixedPointV0(atoms="1", scale=scale)


def test_fixed_point_rejects_binary_float_directly() -> None:
    FixedPointV0 = _contracts().FixedPointV0

    with pytest.raises(ValueError, match="binary float"):
        FixedPointV0.from_value(0.5)


@pytest.mark.parametrize("value", ["1e3", "1E-3", Decimal("1E+3")])
def test_fixed_point_rejects_exponent_notation(value: str | Decimal) -> None:
    FixedPointV0 = _contracts().FixedPointV0

    with pytest.raises(ValueError, match="exponent"):
        FixedPointV0.from_value(value)


def test_fixed_point_conversion_is_exact_and_preserves_declared_scale() -> None:
    FixedPointV0 = _contracts().FixedPointV0

    exact = FixedPointV0.from_value("-12345678901234567890.00120")

    assert exact == FixedPointV0(atoms="-1234567890123456789000120", scale=5)
    assert exact.to_decimal() == Decimal("-12345678901234567890.00120")
    assert FixedPointV0.from_value(7) == FixedPointV0(atoms="7", scale=0)
    assert FixedPointV0.from_value(Decimal("0.125")) == FixedPointV0(
        atoms="125", scale=3
    )


@pytest.mark.parametrize(
    "bad_timestamp",
    [
        "2026-07-22T12:00:00+00:00",
        "2026-07-22T07:00:00-05:00",
        "2026-07-22 12:00:00Z",
        "2026-07-22T12:00:00z",
        "2026-13-99T25:61:61Z",
    ],
)
def test_event_rejects_noncanonical_utc_timestamp(bad_timestamp: str) -> None:
    EventEnvelopeV0 = _contracts().EventEnvelopeV0
    event = _raw_event()
    event["time"]["receive_at"] = bad_timestamp

    with pytest.raises(ValidationError):
        EventEnvelopeV0.model_validate(event)


@pytest.mark.parametrize(
    "field,bad_hash",
    [
        ("payload_sha256", "sha256:" + "A" * 64),
        ("payload_sha256", "0" * 64),
        ("rule_snapshot_ref", "sha256:" + "g" * 64),
    ],
)
def test_event_rejects_noncanonical_sha256(field: str, bad_hash: str) -> None:
    EventEnvelopeV0 = _contracts().EventEnvelopeV0
    event = _derived_event()
    event[field] = bad_hash

    with pytest.raises(ValidationError):
        EventEnvelopeV0.model_validate(event)


def test_event_forbids_unknown_top_level_and_processing_time_fields() -> None:
    EventEnvelopeV0 = _contracts().EventEnvelopeV0
    event = _raw_event()
    event["unexpected"] = "not canonical"
    event["time"]["processing_at"] = "2026-07-22T12:00:02Z"

    with pytest.raises(ValidationError) as error:
        EventEnvelopeV0.model_validate(event)

    locations = {tuple(item["loc"]) for item in error.value.errors()}
    assert ("unexpected",) in locations
    assert ("time", "processing_at") in locations


@pytest.mark.parametrize(
    "payload",
    [
        {"price": 0.5},
        {"levels": [{"price": {"atoms": "1", "scale": 1}, "size": 2.5}]},
        {"nested": {"deeper": [1, 2, {"value": float("nan")}]}},
    ],
)
def test_event_rejects_binary_float_anywhere_in_payload(payload: dict[str, Any]) -> None:
    EventEnvelopeV0 = _contracts().EventEnvelopeV0
    event = _raw_event()
    event["payload"] = payload
    event["payload_sha256"] = _digest("f")

    with pytest.raises(ValidationError, match="binary float"):
        EventEnvelopeV0.model_validate(event)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda event: event["source"].update(capture_session_id=None),
        lambda event: event["source"].update(record_ordinal=None),
        lambda event: event.update(native_refs=[]),
        lambda event: event.update(lineage={"parent_event_ids": [_event_id("a")]}),
        lambda event: event["lineage"].update(raw_record_ordinal=24),
    ],
)
def test_raw_observation_requires_capture_native_and_matching_raw_lineage(
    mutation,
) -> None:
    EventEnvelopeV0 = _contracts().EventEnvelopeV0
    event = _raw_event()
    mutation(event)

    with pytest.raises(ValidationError):
        EventEnvelopeV0.model_validate(event)


@pytest.mark.parametrize(
    "event_type",
    [
        "normalized_observation",
        "model_output",
        "label",
        "signal",
        "simulated_order",
        "simulated_fill",
        "simulated_pnl",
    ],
)
def test_derived_events_require_parent_ids_and_registered_experiment(
    event_type: str,
) -> None:
    EventEnvelopeV0 = _contracts().EventEnvelopeV0
    event = _derived_event()
    event["event_type"] = event_type
    event["lineage"] = {"parent_event_ids": []}
    event["experiment_id"] = None

    with pytest.raises(ValidationError):
        EventEnvelopeV0.model_validate(event)


@pytest.mark.parametrize("experiment_id", ["X-00", "X-1", "X-11", "R-001", "x-01"])
def test_event_rejects_unregistered_experiment_id(experiment_id: str) -> None:
    EventEnvelopeV0 = _contracts().EventEnvelopeV0
    event = _derived_event()
    event["experiment_id"] = experiment_id

    with pytest.raises(ValidationError):
        EventEnvelopeV0.model_validate(event)


def test_event_rejects_payload_hash_mismatch() -> None:
    EventEnvelopeV0 = _contracts().EventEnvelopeV0
    event = _raw_event()
    event["payload"]["asset_id"] = "mutated"

    with pytest.raises(ValidationError, match="payload_sha256"):
        EventEnvelopeV0.model_validate(event)


def test_event_rejects_nondeterministic_event_id() -> None:
    EventEnvelopeV0 = _contracts().EventEnvelopeV0
    event = _raw_event()
    event["event_id"] = _event_id("9")

    with pytest.raises(ValidationError, match="event_id"):
        EventEnvelopeV0.model_validate(event)


def test_event_id_hashes_full_model_dump_excluding_only_event_id() -> None:
    contracts = _contracts()
    envelope = contracts.EventEnvelopeV0.model_validate(_derived_event())
    material = envelope.model_dump(mode="python")
    material.pop("event_id")
    documented = "evt_" + hashlib.sha256(
        contracts.canonical_json_bytes(material)
    ).hexdigest()

    assert contracts.event_id_for(envelope) == documented


def test_raw_event_may_omit_optional_experiment_and_rule_snapshot_refs() -> None:
    contracts = _contracts()
    event = _raw_event()
    event.pop("experiment_id")
    event.pop("rule_snapshot_ref")
    event["event_id"] = contracts.event_id_for(event)

    envelope = contracts.EventEnvelopeV0.model_validate(event)

    assert envelope.experiment_id is None
    assert envelope.rule_snapshot_ref is None


@pytest.mark.parametrize(
    "event_type", ["simulated_order", "simulated_fill", "simulated_pnl"]
)
def test_simulated_events_require_rule_snapshot_ref(event_type: str) -> None:
    contracts = _contracts()
    event = _derived_event()
    event["event_type"] = event_type
    event["rule_snapshot_ref"] = None
    event["event_id"] = contracts.event_id_for(event)

    with pytest.raises(ValidationError, match="rule_snapshot_ref"):
        contracts.EventEnvelopeV0.model_validate(event)


@pytest.mark.parametrize(
    "event_type", ["simulated_order", "simulated_fill", "simulated_pnl"]
)
def test_simulated_events_accept_observed_rule_snapshot_ref(event_type: str) -> None:
    contracts = _contracts()
    event = _derived_event()
    event["event_type"] = event_type
    event["event_id"] = contracts.event_id_for(event)

    envelope = contracts.EventEnvelopeV0.model_validate(event)

    assert envelope.rule_snapshot_ref == _digest("c")


def test_event_serialization_is_deterministic_for_set_semantics() -> None:
    contracts = _contracts()
    first = _derived_event()
    second = copy.deepcopy(first)
    second["quality_flags"].reverse()
    second["lineage"]["parent_event_ids"].reverse()
    second["canonical_refs"]["participant_ids"].reverse()
    second["event_id"] = contracts.event_id_for(second)

    first_model = contracts.EventEnvelopeV0.model_validate(first)
    second_model = contracts.EventEnvelopeV0.model_validate(second)

    assert first_model.event_id == second_model.event_id
    assert contracts.canonical_json_bytes(first_model) == contracts.canonical_json_bytes(
        second_model
    )
    assert contracts.canonical_sha256(first_model) == contracts.canonical_sha256(
        second_model
    )


def test_event_accepts_valid_raw_and_derived_contracts() -> None:
    EventEnvelopeV0 = _contracts().EventEnvelopeV0

    raw = EventEnvelopeV0.model_validate(_raw_event())
    derived = EventEnvelopeV0.model_validate(_derived_event())

    assert raw.event_id.startswith("evt_")
    assert derived.experiment_id == "X-06"


def test_model_output_is_complete_state_transition_distribution() -> None:
    ModelOutputV0 = _contracts().ModelOutputV0

    output = ModelOutputV0.model_validate(_model_output())

    assert set(output.probabilities) == set(output.state_space)
    assert sum(
        (probability.to_decimal() for probability in output.probabilities.values()),
        start=Decimal(0),
    ) == Decimal(1)


@pytest.mark.parametrize(
    "probabilities",
    [
        {},
        {
            "away_possession": {"atoms": "25", "scale": 2},
            "home_possession": {"atoms": "50", "scale": 2},
            "game_over": {"atoms": "24", "scale": 2},
        },
        {
            "away_possession": {"atoms": "-1", "scale": 2},
            "home_possession": {"atoms": "76", "scale": 2},
            "game_over": {"atoms": "25", "scale": 2},
        },
        {
            "away_possession": {"atoms": "25", "scale": 2},
            "home_possession": {"atoms": "50", "scale": 2},
            "wrong_state": {"atoms": "25", "scale": 2},
        },
    ],
)
def test_model_output_rejects_invalid_probability_distribution(
    probabilities: dict[str, Any],
) -> None:
    ModelOutputV0 = _contracts().ModelOutputV0
    output = _model_output()
    output["probabilities"] = probabilities

    with pytest.raises(ValidationError):
        ModelOutputV0.model_validate(output)


def test_model_output_rejects_final_win_probability_shortcut() -> None:
    ModelOutputV0 = _contracts().ModelOutputV0
    output = _model_output()
    output.pop("state_space")
    output.pop("probabilities")
    output["win_probability"] = {"atoms": "5", "scale": 1}

    with pytest.raises(ValidationError):
        ModelOutputV0.model_validate(output)


def test_venue_rule_snapshot_requires_observed_per_market_values() -> None:
    VenueRuleSnapshotV0 = _contracts().VenueRuleSnapshotV0

    snapshot = VenueRuleSnapshotV0.model_validate(_rule_snapshot())

    assert snapshot.condition_id.startswith("condition_")
    assert snapshot.seconds_delay.to_decimal() == Decimal(1)
    assert snapshot.raw_response_hash.startswith("sha256:")


@pytest.mark.parametrize(
    "missing_field",
    [
        "condition_id",
        "seconds_delay",
        "cancel_during_delay",
        "start_time_cancel_policy",
        "fees_enabled",
        "fee_rate",
        "fee_exponent",
        "taker_only",
        "maker_fee_rate",
        "minimum_tick_size",
        "minimum_order_size",
        "order_types_supported",
        "source_document_version",
        "raw_response_hash",
    ],
)
def test_venue_rule_snapshot_never_defaults_missing_execution_values(
    missing_field: str,
) -> None:
    VenueRuleSnapshotV0 = _contracts().VenueRuleSnapshotV0
    snapshot = _rule_snapshot()
    snapshot.pop(missing_field)

    with pytest.raises(ValidationError):
        VenueRuleSnapshotV0.model_validate(snapshot)


@pytest.mark.parametrize(
    "field,value",
    [
        ("venue", "unknown"),
        ("condition_id", "O-002"),
        ("seconds_delay", {"atoms": "-1", "scale": 0}),
        ("start_time_cancel_policy", "unknown"),
        ("fee_rate", {"atoms": "-1", "scale": 2}),
        ("minimum_tick_size", {"atoms": "0", "scale": 0}),
        ("minimum_order_size", {"atoms": "0", "scale": 0}),
        ("order_types_supported", []),
        ("order_types_supported", ["UNKNOWN"]),
        ("source_document_version", "unknown"),
        ("raw_response_hash", "sha256:" + "A" * 64),
    ],
)
def test_venue_rule_snapshot_fails_closed_on_unknown_or_invalid_values(
    field: str, value: Any
) -> None:
    VenueRuleSnapshotV0 = _contracts().VenueRuleSnapshotV0
    snapshot = _rule_snapshot()
    snapshot[field] = value

    with pytest.raises(ValidationError):
        VenueRuleSnapshotV0.model_validate(snapshot)


def test_venue_is_not_a_global_constant_and_conflicting_observations_survive() -> None:
    VenueRuleSnapshotV0 = _contracts().VenueRuleSnapshotV0
    first = _rule_snapshot()
    second = copy.deepcopy(first)
    second["venue"] = "kalshi"
    second["condition_id"] = "condition_kalshi_game_001"
    second["seconds_delay"] = {"atoms": "3", "scale": 0}
    second["raw_response_hash"] = _digest("f")

    snapshots = [
        VenueRuleSnapshotV0.model_validate(first),
        VenueRuleSnapshotV0.model_validate(second),
    ]

    assert [snapshot.venue for snapshot in snapshots] == ["polymarket", "kalshi"]
    assert [snapshot.seconds_delay.to_decimal() for snapshot in snapshots] == [
        Decimal(1),
        Decimal(3),
    ]


def test_controlled_python_enums_are_exact() -> None:
    contracts = _contracts()

    assert contracts.MARKET_RELATIONS == frozenset(
        {
            "identity",
            "subset",
            "superset",
            "overlap",
            "mutex",
            "exhaustive",
            "incompatible",
        }
    )
    assert contracts.REGISTERED_EXPERIMENT_IDS == frozenset(
        f"X-{number:02d}" for number in range(1, 11)
    )
    assert "gap_detected" in contracts.QUALITY_FLAGS
    assert "unknown" not in contracts.QUALITY_FLAGS


SCHEMA_PATHS = (
    "event-envelope/v0.schema.yaml",
    "id-registry/v0/entity.schema.yaml",
    "id-registry/v0/native-assertion.schema.yaml",
    "id-registry/v0/relation-assertion.schema.yaml",
    "model-output/v0.schema.yaml",
    "quality-flags/v0.yaml",
    "market-relations/v0.yaml",
    "venue-rule-snapshot/v0.schema.yaml",
)


@pytest.mark.parametrize("relative_path", SCHEMA_PATHS)
def test_machine_readable_contract_documents_load(relative_path: str) -> None:
    path = CONTRACTS_ROOT / relative_path
    assert path.is_file(), f"missing contract: {relative_path}"

    document = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(document, dict)
    assert document.get("contract_version") == "v0"
    assert document.get("title")


def test_schema_contracts_forbid_extensions_and_require_core_keys() -> None:
    expected_required = {
        "event-envelope/v0.schema.yaml": {
            "envelope_version",
            "event_id",
            "event_type",
            "payload_schema_version",
            "source",
            "time",
            "canonical_refs",
            "native_refs",
            "lineage",
            "quality_flags",
            "payload",
            "payload_sha256",
        },
        "model-output/v0.schema.yaml": {
            "contract_version",
            "model_id",
            "model_version",
            "experiment_id",
            "run_id",
            "game_id",
            "state_event_id",
            "pit_cutoff_at",
            "state_space",
            "horizon",
            "probabilities",
            "feature_sha256",
            "data_sha256",
            "config_sha256",
            "quality_flags",
        },
        "venue-rule-snapshot/v0.schema.yaml": set(_rule_snapshot()),
    }

    for relative_path, required in expected_required.items():
        document = yaml.safe_load(
            (CONTRACTS_ROOT / relative_path).read_text(encoding="utf-8")
        )
        assert document["type"] == "object"
        assert document["additionalProperties"] is False
        assert set(document["required"]) == required
        assert required <= set(document["properties"])


def test_id_registry_assertions_separate_catalog_and_domain_ids() -> None:
    for filename in (
        "entity.schema.yaml",
        "native-assertion.schema.yaml",
        "relation-assertion.schema.yaml",
    ):
        document = yaml.safe_load(
            (CONTRACTS_ROOT / "id-registry" / "v0" / filename).read_text(
                encoding="utf-8"
            )
        )
        assert document["catalog_ids_are_domain_ids"] is False
        assert set(document["catalog_id_namespaces"]) == {"R", "I", "O"}
        assert set(document["domain_id_namespaces"]) == {
            "competition",
            "game",
            "participant",
            "venue_event",
            "market",
            "outcome",
            "condition",
        }


def _id_pair_matches_schema(
    document: dict[str, Any], entity_type: str, canonical_id: str
) -> bool:
    matches = 0
    for branch in document["oneOf"]:
        properties = branch["properties"]
        expected_type = properties["entity_type"]["const"]
        id_pattern = properties["canonical_id"]["pattern"]
        if entity_type == expected_type and re.fullmatch(id_pattern, canonical_id):
            matches += 1
    return matches == 1


@pytest.mark.parametrize(
    "filename", ["entity.schema.yaml", "native-assertion.schema.yaml"]
)
def test_id_assertions_enforce_entity_type_and_canonical_prefix_pairs(
    filename: str,
) -> None:
    document = yaml.safe_load(
        (CONTRACTS_ROOT / "id-registry" / "v0" / filename).read_text(
            encoding="utf-8"
        )
    )
    valid_pairs = {
        "competition": "cmp_nba",
        "game": "game_nba_2026_001",
        "participant": "participant_home",
        "venue_event": "venue_event_pm_001",
        "market": "market_moneyline_001",
        "outcome": "outcome_home_win",
        "condition": "condition_0xabc",
    }

    for entity_type, canonical_id in valid_pairs.items():
        assert _id_pair_matches_schema(document, entity_type, canonical_id)

    assert not _id_pair_matches_schema(document, "game", "participant_home")
    assert not _id_pair_matches_schema(document, "competition", "competition_nba")
    assert not _id_pair_matches_schema(document, "condition", "O-002")


def test_event_schema_and_python_agree_optional_refs_and_simulated_rule_gate() -> None:
    contracts = _contracts()
    document = yaml.safe_load(
        (CONTRACTS_ROOT / "event-envelope" / "v0.schema.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert "experiment_id" not in document["required"]
    assert "rule_snapshot_ref" not in document["required"]
    assert not contracts.EventEnvelopeV0.model_fields["experiment_id"].is_required()
    assert not contracts.EventEnvelopeV0.model_fields["rule_snapshot_ref"].is_required()

    conditional_types = {
        event_type
        for clause in document["allOf"]
        for event_type in clause.get("if", {})
        .get("properties", {})
        .get("event_type", {})
        .get("enum", [])
        if "rule_snapshot_ref" in clause.get("then", {}).get("required", [])
    }
    assert conditional_types == {"simulated_order", "simulated_fill", "simulated_pnl"}


def test_event_schema_declares_defaults_used_by_full_model_event_hash() -> None:
    document = yaml.safe_load(
        (CONTRACTS_ROOT / "event-envelope" / "v0.schema.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert document["properties"]["experiment_id"]["default"] is None
    assert document["properties"]["rule_snapshot_ref"]["default"] is None
    for field in ("venue", "sequence", "capture_session_id", "record_ordinal"):
        assert document["$defs"]["source"]["properties"][field]["default"] is None
    for field in ("source_at", "publish_at", "exchange_at"):
        assert document["$defs"]["time"]["properties"][field]["default"] is None
    assert document["$defs"]["raw_lineage"]["properties"]["parent_event_ids"][
        "default"
    ] == []
    assert document["$defs"]["parent_lineage"]["properties"]["raw_object_hash"][
        "default"
    ] is None
    assert document["$defs"]["parent_lineage"]["properties"][
        "raw_record_ordinal"
    ]["default"] is None


def test_quality_and_relation_yaml_enums_match_python_contract() -> None:
    contracts = _contracts()
    quality = yaml.safe_load(
        (CONTRACTS_ROOT / "quality-flags" / "v0.yaml").read_text(encoding="utf-8")
    )
    relations = yaml.safe_load(
        (CONTRACTS_ROOT / "market-relations" / "v0.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert frozenset(quality["enum"]) == contracts.QUALITY_FLAGS
    assert frozenset(relations["enum"]) == contracts.MARKET_RELATIONS


def test_deterministic_replay_document_records_level_1_and_2_gate() -> None:
    path = CONTRACTS_ROOT / "deterministic-replay" / "v0.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8").lower()

    for clause in (
        "level 1",
        "level 2",
        "level 3",
        "receive_at",
        "source_at",
        "market_id",
        "outcome_id",
        "payload_sha256",
        "sha-256",
        "fixed-point",
        "utc",
        "explicit order by",
        "single writer",
        "lockfile",
        "random seed",
        "nulls last",
        "uint64be",
        "32-byte",
        "domain tag",
        "rule_snapshot_ref",
    ):
        assert clause in text
    assert "level 1 + level 2" in text
    assert "wall-clock processing" not in text


@pytest.mark.parametrize(
    "field_path",
    [
        ("time", "source_at"),
        ("canonical_refs", "market_id"),
        ("canonical_refs", "outcome_id"),
    ],
)
def test_replay_order_key_places_every_nullable_field_last(
    field_path: tuple[str, str],
) -> None:
    contracts = _contracts()
    present = _derived_event()
    null_value = copy.deepcopy(present)
    parent, child = field_path
    null_value[parent][child] = None

    assert contracts.replay_order_key(present) < contracts.replay_order_key(null_value)


def test_replay_order_key_uses_documented_tagged_null_sentinel() -> None:
    contracts = _contracts()
    event = _derived_event()
    event["time"]["source_at"] = None
    event["canonical_refs"]["market_id"] = None
    event["canonical_refs"]["outcome_id"] = None

    key = contracts.replay_order_key(event)

    assert key[1:4] == ((1, ""), (1, ""), (1, ""))


def test_level_2_stream_hash_uses_explicit_domain_count_and_digest_framing() -> None:
    contracts = _contracts()
    event_ids = [_event_id("a"), _event_id("b")]
    expected_frame = (
        contracts.LEVEL2_STREAM_DOMAIN_TAG
        + len(event_ids).to_bytes(8, byteorder="big", signed=False)
        + bytes.fromhex("a" * 64)
        + bytes.fromhex("b" * 64)
    )

    frame = contracts.level2_stream_frame(event_ids)

    assert frame == expected_frame
    assert len(frame) == len(contracts.LEVEL2_STREAM_DOMAIN_TAG) + 8 + 2 * 32
    assert contracts.level2_stream_sha256(event_ids) == (
        "sha256:" + hashlib.sha256(expected_frame).hexdigest()
    )
    assert contracts.level2_stream_sha256(reversed(event_ids)) != (
        contracts.level2_stream_sha256(event_ids)
    )


@pytest.mark.parametrize(
    "filename,required_phrases",
    [
        (
            "0001-engine-decision-deferred.md",
            ("accepted", "x-09", "deferred", "evidence", "no engine selected"),
        ),
        (
            "0002-native-adapter-control-plane-boundary.md",
            ("accepted", "native adapter", "control plane", "wire protocol"),
        ),
        (
            "0003-hot-path-boundary.md",
            ("accepted", "hot path", "pmxt", "llm", "slow path"),
        ),
        (
            "0004-event-sourcing.md",
            ("accepted", "append-only", "event sourcing", "reconstruct"),
        ),
        (
            "0005-fail-closed.md",
            ("accepted", "fail closed", "unknown order", "unknown rule", "gap"),
        ),
    ],
)
def test_adrs_record_approved_decisions_and_no_go(
    filename: str, required_phrases: tuple[str, ...]
) -> None:
    path = ADR_ROOT / filename
    assert path.is_file(), f"missing ADR: {filename}"
    text = path.read_text(encoding="utf-8").lower()

    for phrase in required_phrases:
        assert phrase in text
    for no_go in (
        "real-money execution",
        "maker queue",
        "multi-venue live arbitrage",
        "copy trading",
        "llm hot path",
        "reinforcement learning",
        "microservices",
    ):
        assert no_go in text
