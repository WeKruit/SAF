from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from prediction_market.contracts import (
    EventEnvelopeV0,
    FixedPointV0,
    VenueRuleSnapshotV0,
    canonical_sha256,
)
from prediction_market.vertical_slice import (
    X09ConfigV0,
    X09ExperimentBlocked,
    X09FixtureV0,
    X09VerticalSliceInputError,
    compute_x09_runtime_hashes,
    run_vertical_slice,
    run_x09_formal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def _sha256(fill: str) -> str:
    return "sha256:" + fill * 64


def _event_id(fill: str) -> str:
    return "evt_" + fill * 64


def _fixed(value: str) -> FixedPointV0:
    return FixedPointV0.from_value(value)


def _timestamp(seconds: str) -> str:
    value = T0 + timedelta(microseconds=int(Decimal(seconds) * Decimal(1_000_000)))
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _level(price: str, quantity: str) -> dict[str, object]:
    return {
        "price": _fixed(price).model_dump(mode="json"),
        "quantity": _fixed(quantity).model_dump(mode="json"),
    }


def _source_event(
    *,
    sequence: int,
    receive_at: str,
    payload: dict[str, object],
) -> EventEnvelopeV0:
    return EventEnvelopeV0.create(
        envelope_version="v0",
        event_type="normalized_observation",
        payload_schema_version="v0",
        source={
            "system": "x01-reconstructor",
            "stream": "frozen-game",
            "venue": "polymarket",
            "sequence": sequence,
            "capture_session_id": None,
            "record_ordinal": None,
        },
        time={
            "receive_at": receive_at,
            "receive_basis": "upstream_exporter",
            "source_at": receive_at,
            "publish_at": None,
            "exchange_at": None,
        },
        canonical_refs={
            "competition_id": "cmp_nba",
            "game_id": "game_nba_x09",
            "participant_ids": ["participant_away", "participant_home"],
            "venue_event_id": "venue_event_x09",
            "market_id": "market_x09",
            "outcome_id": "outcome_home",
            "condition_id": "condition_x09",
        },
        native_refs=[],
        lineage={"parent_event_ids": [_event_id(format(sequence, "x")[-1])]},
        experiment_id="X-01",
        rule_snapshot_ref=None,
        quality_flags=[],
        payload=payload,
    )


def _with_experiment(
    event: EventEnvelopeV0, experiment_id: str
) -> EventEnvelopeV0:
    material = event.model_dump(mode="python", round_trip=True)
    material.pop("event_id")
    material.pop("payload_sha256")
    material["experiment_id"] = experiment_id
    return EventEnvelopeV0.create(**material)


def _rule_snapshot() -> VenueRuleSnapshotV0:
    return VenueRuleSnapshotV0.model_validate(
        {
            "venue": "polymarket",
            "condition_id": "condition_x09",
            "fetched_at": "2026-07-22T11:59:00Z",
            "effective_from": "2026-07-22T11:59:00Z",
            "game_start_time": "2026-07-22T13:00:00Z",
            "seconds_delay": _fixed("1").model_dump(mode="json"),
            "cancel_during_delay": False,
            "start_time_cancel_policy": (
                "cancel_all_with_schedule_change_exception"
            ),
            "fees_enabled": True,
            "fee_rate": _fixed("0.02").model_dump(mode="json"),
            "fee_exponent": _fixed("1").model_dump(mode="json"),
            "taker_only": True,
            "maker_fee_rate": _fixed("0").model_dump(mode="json"),
            "minimum_tick_size": _fixed("0.01").model_dump(mode="json"),
            "minimum_order_size": _fixed("1").model_dump(mode="json"),
            "order_types_supported": ["MARKET"],
            "source_document_version": "x09-fixture@2026-07-22",
            "raw_response_hash": _sha256("9"),
        }
    )


def _fixture() -> X09FixtureV0:
    score = _source_event(
        sequence=1,
        receive_at=_timestamp("0"),
        payload={"kind": "score", "points": 2, "team": "home"},
    )
    preeligible = _source_event(
        sequence=2,
        receive_at=_timestamp("5.5"),
        payload={
            "kind": "book",
            "suspended": False,
            "bids": [_level("0.09", "10")],
            "asks": [_level("0.10", "10")],
        },
    )
    execution = _source_event(
        sequence=3,
        receive_at=_timestamp("6"),
        payload={
            "kind": "book",
            "suspended": False,
            "bids": [_level("0.48", "10")],
            "asks": [_level("0.50", "3"), _level("0.55", "4")],
        },
    )
    mark = _source_event(
        sequence=4,
        receive_at=_timestamp("10"),
        payload={
            "kind": "book",
            "suspended": False,
            "bids": [_level("0.60", "10")],
            "asks": [_level("0.62", "10")],
        },
    )
    config = X09ConfigV0(
        order_quantity=_fixed("5"),
        maximum_order_quantity=_fixed("5"),
        own_delay=timedelta(0),
        pnl_horizon=timedelta(seconds=4),
        random_seed=0,
        canonical_id_snapshot_sha256=_sha256("c"),
        dependency_lock_sha256=_sha256("d"),
    )
    # The fixture deliberately does not depend on caller order.
    return X09FixtureV0(
        events=(mark, execution, score, preeligible),
        rule_snapshot=_rule_snapshot(),
        config=config,
    )


def _as_decimal(payload: object) -> Decimal:
    assert isinstance(payload, Mapping)
    return FixedPointV0.model_validate(dict(payload)).to_decimal()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


def test_vertical_slice_has_identical_semantics_and_hash_twice() -> None:
    first = run_vertical_slice(_fixture())
    second = run_vertical_slice(_fixture())

    assert first.harness_status == "HARNESS_PASS"
    assert first.experiment_status == "EXPERIMENT_BLOCKED"
    assert first.first_run.semantic_summary == first.second_run.semantic_summary
    assert first.first_run.stream_sha256 == first.second_run.stream_sha256
    assert first.semantic_summary == second.semantic_summary
    assert first.stream_sha256 == second.stream_sha256
    assert first.canonical_log == second.canonical_log
    assert first.stream_sha256.startswith("sha256:")
    assert len(first.stream_sha256) == 71
    assert len(first.orders) == len(first.fills) == len(first.pnl_events) == 1


def test_signal_is_five_seconds_after_score_and_order_names_trigger_lineage() -> None:
    result = run_vertical_slice(_fixture())
    score = next(event for event in result.events if event.payload.get("kind") == "score")
    signal = next(event for event in result.events if event.event_type == "signal")
    order = result.orders[0]

    assert _parse_timestamp(signal.time.receive_at) - _parse_timestamp(
        score.time.receive_at
    ) == timedelta(seconds=5)
    assert signal.lineage.parent_event_ids == (score.event_id,)
    assert set(order.lineage.parent_event_ids) == {score.event_id, signal.event_id}
    assert order.payload["trigger_event_id"] == score.event_id


def test_x09_taker_fill_and_pnl_use_rule_snapshot_and_exact_fixed_point() -> None:
    fixture = _fixture()
    result = run_vertical_slice(fixture)
    fill = result.fills[0]
    pnl = result.pnl_events[0]
    expected_rule_ref = canonical_sha256(
        fixture.rule_snapshot.model_dump(mode="json", round_trip=True)
    )

    simulated = result.orders + result.fills + result.pnl_events
    assert all(event.experiment_id == "X-09" for event in simulated)
    assert all(
        event.rule_snapshot_ref == expected_rule_ref
        for event in simulated
    )
    assert fill.time.receive_at == _timestamp("6")
    assert _as_decimal(fill.payload["vwap"]) == Decimal("0.52")
    assert _as_decimal(fill.payload["gross_cost"]) == Decimal("2.60")
    assert _as_decimal(fill.payload["fee"]) == Decimal("0.02496")
    assert _as_decimal(fill.payload["total_cost"]) == Decimal("2.62496")
    assert _as_decimal(pnl.payload["gross_proceeds"]) == Decimal("3.00")
    assert _as_decimal(pnl.payload["pnl"]) == Decimal("0.37504")


def test_risk_limit_rejects_order_before_any_simulated_event() -> None:
    fixture = _fixture()
    rejected = replace(
        fixture,
        config=replace(fixture.config, order_quantity=_fixed("6")),
    )

    with pytest.raises(X09VerticalSliceInputError, match="risk limit"):
        run_vertical_slice(rejected)


def test_vertical_slice_rejects_input_outside_the_x01_stream() -> None:
    fixture = _fixture()
    foreign = _with_experiment(fixture.events[0], "X-09")
    invalid = replace(fixture, events=(foreign, *fixture.events[1:]))

    with pytest.raises(X09VerticalSliceInputError, match="X-01"):
        run_vertical_slice(invalid)


def test_runtime_hashes_bind_the_exact_fixture_configuration() -> None:
    fixture = _fixture()
    changed = replace(
        fixture,
        config=replace(fixture.config, random_seed=1),
    )

    first = compute_x09_runtime_hashes(fixture)
    second = compute_x09_runtime_hashes(changed)

    assert first.code_sha256 == second.code_sha256
    assert first.data_sha256 != second.data_sha256


def test_formal_entry_reads_registry_and_reports_current_blockers() -> None:
    with pytest.raises(X09ExperimentBlocked) as captured:
        run_x09_formal(PROJECT_ROOT, _fixture())

    assert any("x09_input_manifest" in blocker for blocker in captured.value.blockers)
    assert any("X-01" in blocker for blocker in captured.value.blockers)
    assert any("preregistered" in blocker for blocker in captured.value.blockers)
