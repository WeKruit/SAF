from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from prediction_market.contracts import FixedPointV0
from prediction_market.labels import (
    BarrierLabelParameters,
    LabelAuthorizationError,
    LabelInputError,
    QuoteV0,
    compute_x05_runtime_hashes,
    generate_x05_long_labels,
    label_long,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 7, 22, 18, 0, tzinfo=UTC)


def _price(value: str) -> FixedPointV0:
    return FixedPointV0.from_value(value)


def _quote(
    quote_id: str,
    seconds: int,
    bid: str | None,
    ask: str | None,
    *,
    state: str = "ACTIVE",
    source_lag_ms: int = 20,
) -> QuoteV0:
    received_at = T0 + timedelta(seconds=seconds)
    return QuoteV0(
        quote_id=quote_id,
        source_at=received_at - timedelta(milliseconds=source_lag_ms),
        received_at=received_at,
        state=state,
        bid=None if bid is None else _price(bid),
        ask=None if ask is None else _price(ask),
    )


def _parameters(
    *,
    same_time_touch_rule: str = "AMBIGUOUS",
    overlap_rule: str = "KEEP_GROUPED",
) -> BarrierLabelParameters:
    return BarrierLabelParameters(
        upper_return=Decimal("0.10"),
        lower_return=Decimal("0.10"),
        horizon=timedelta(seconds=60),
        max_quote_age=timedelta(milliseconds=100),
        same_time_touch_rule=same_time_touch_rule,
        overlap_rule=overlap_rule,
        purge=timedelta(seconds=10),
        embargo=timedelta(seconds=5),
    )


def test_long_label_enters_at_ask_and_exits_at_bid() -> None:
    quotes = [
        _quote("q-entry", 0, "0.48", "0.50"),
        _quote("q-hit", 5, "0.56", "0.58"),
    ]

    label = label_long(quotes, anchor_at=T0, parameters=_parameters())

    assert label.entry_price == _price("0.50")
    assert label.exit_price == _price("0.56")
    assert label.entry_quote_id == "q-entry"
    assert label.exit_quote_id == "q-hit"
    assert label.outcome == "UPPER"


def test_midpoint_is_never_an_entry_or_exit_price() -> None:
    quotes = [
        _quote("q-entry", 0, "0.46", "0.50"),
        _quote("q-hit", 5, "0.56", "0.60"),
    ]

    label = label_long(quotes, anchor_at=T0, parameters=_parameters())

    assert label.entry_price.to_decimal() != Decimal("0.48")
    assert label.exit_price.to_decimal() != Decimal("0.58")


def test_suspension_uses_first_nonstale_executable_quote_after_resume() -> None:
    quotes = [
        _quote("suspend", 0, None, None, state="SUSPENDED"),
        _quote("stale", 1, "0.49", "0.51", source_lag_ms=101),
        _quote("resume", 2, "0.50", "0.52"),
        _quote("horizon", 62, "0.51", "0.53"),
    ]

    label = label_long(quotes, anchor_at=T0, parameters=_parameters())

    assert label.entry_quote_id == "resume"
    assert label.resume_quote_ids == ("resume",)
    assert label.entry_price == _price("0.52")
    assert label.exit_quote_id == "horizon"
    assert label.exit_price == _price("0.51")
    assert label.outcome == "HORIZON"


def test_same_timestamp_suspension_blocks_active_entry_for_whole_group() -> None:
    quotes = [
        _quote("active-same-time", 0, "0.49", "0.51"),
        _quote("suspend-same-time", 0, None, None, state="SUSPENDED"),
        _quote("resume", 1, "0.50", "0.52"),
        _quote("horizon", 61, "0.51", "0.53"),
    ]

    label = label_long(quotes, anchor_at=T0, parameters=_parameters())

    assert label.entry_quote_id == "resume"
    assert label.resume_quote_ids == ("resume",)


def test_same_timestamp_quote_after_entry_can_touch_barrier() -> None:
    quotes = [
        _quote("a-entry", 0, "0.49", "0.50"),
        _quote("b-upper", 0, "0.56", "0.58"),
        _quote("horizon", 60, "0.50", "0.52"),
    ]

    label = label_long(quotes, anchor_at=T0, parameters=_parameters())

    assert label.entry_quote_id == "a-entry"
    assert label.exit_quote_id == "b-upper"
    assert label.outcome == "UPPER"


@pytest.mark.parametrize(
    ("rule", "expected"),
    [("UPPER_FIRST", "UPPER"), ("LOWER_FIRST", "LOWER"), ("AMBIGUOUS", "AMBIGUOUS")],
)
def test_same_time_opposite_touches_follow_preregistered_rule(
    rule: str,
    expected: str,
) -> None:
    quotes = [
        _quote("entry", 0, "0.49", "0.50"),
        _quote("lower", 5, "0.44", "0.46"),
        _quote("upper", 5, "0.56", "0.58"),
    ]

    label = label_long(
        quotes,
        anchor_at=T0,
        parameters=_parameters(same_time_touch_rule=rule),
    )

    assert label.outcome == expected
    if expected == "UPPER":
        assert label.exit_quote_id == "upper"
    elif expected == "LOWER":
        assert label.exit_quote_id == "lower"
    else:
        assert label.exit_quote_id is None
        assert label.exit_price is None


def test_parameters_have_no_defaults_and_reject_binary_float() -> None:
    with pytest.raises(TypeError):
        BarrierLabelParameters()  # type: ignore[call-arg]
    with pytest.raises(LabelInputError, match="Decimal"):
        BarrierLabelParameters(
            upper_return=0.1,  # type: ignore[arg-type]
            lower_return=Decimal("0.10"),
            horizon=timedelta(seconds=60),
            max_quote_age=timedelta(milliseconds=100),
            same_time_touch_rule="AMBIGUOUS",
            overlap_rule="KEEP_GROUPED",
            purge=timedelta(0),
            embargo=timedelta(0),
        )


def test_invalid_quote_and_unsorted_duplicate_ids_fail_closed() -> None:
    with pytest.raises(LabelInputError, match="bid.*ask"):
        _quote("crossed", 0, "0.60", "0.50")

    duplicate = _quote("same", 0, "0.49", "0.50")
    with pytest.raises(LabelInputError, match="duplicate quote_id"):
        label_long([duplicate, duplicate], anchor_at=T0, parameters=_parameters())


def test_x05_generation_is_blocked_by_current_registry() -> None:
    quotes = [
        _quote("entry", 0, "0.49", "0.50"),
        _quote("horizon", 60, "0.49", "0.50"),
    ]

    with pytest.raises(LabelAuthorizationError, match="execution"):
        generate_x05_long_labels(
            PROJECT_ROOT,
            quotes=quotes,
            anchors=[T0],
            parameters=_parameters(),
        )


def test_authorized_x05_run_is_bound_to_preregistered_runtime_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prediction_market.labels as labels_module

    quotes = [
        _quote("entry", 0, "0.49", "0.50"),
        _quote("horizon", 60, "0.49", "0.50"),
    ]
    parameters = _parameters()
    hashes = compute_x05_runtime_hashes(quotes, [T0], parameters)
    locks = [
        {
            "id": lock_id,
            "status": "resolved",
            "evidence_ref": hashes.data_sha256
            if lock_id == "x05_quote_manifest"
            else hashes.configuration_sha256,
        }
        for lock_id in (
            "x05_quote_manifest",
            "barrier_values",
            "purge_and_embargo",
            "post_resume_quote_rule",
            "same_time_touch_rule",
        )
    ]
    fake_registry = {
        "X-05": {
            "execution_authorized": True,
            "authorization_scopes": {
                "label_generation": {
                    "authorized": True,
                    "required_lock_ids": [lock["id"] for lock in locks],
                }
            },
            "registration_locks": locks,
            "preregistered_inputs": {
                "label_generation": {
                    "code_sha256": hashes.code_sha256,
                    "data_sha256": hashes.data_sha256,
                }
            },
            "dependencies": [],
        }
    }
    monkeypatch.setattr(labels_module, "load_experiment_registry", lambda _: fake_registry)

    changed_parameters = BarrierLabelParameters(
        upper_return=Decimal("0.11"),
        lower_return=parameters.lower_return,
        horizon=parameters.horizon,
        max_quote_age=parameters.max_quote_age,
        same_time_touch_rule=parameters.same_time_touch_rule,
        overlap_rule=parameters.overlap_rule,
        purge=parameters.purge,
        embargo=parameters.embargo,
    )
    with pytest.raises(LabelAuthorizationError, match="configuration hash"):
        generate_x05_long_labels(
            PROJECT_ROOT,
            quotes=quotes,
            anchors=[T0],
            parameters=changed_parameters,
        )

    changed_quotes = [
        _quote("entry", 0, "0.48", "0.50"),
        _quote("horizon", 60, "0.49", "0.50"),
    ]
    with pytest.raises(LabelAuthorizationError, match="data hash"):
        generate_x05_long_labels(
            PROJECT_ROOT,
            quotes=changed_quotes,
            anchors=[T0],
            parameters=parameters,
        )

    with pytest.raises(LabelAuthorizationError, match="data hash"):
        generate_x05_long_labels(
            PROJECT_ROOT,
            quotes=quotes,
            anchors=[T0 + timedelta(seconds=1)],
            parameters=parameters,
        )


def test_generator_freezes_quote_sequence_before_hash_and_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prediction_market.labels as labels_module

    first = [
        _quote("entry", 0, "0.49", "0.50"),
        _quote("horizon", 60, "0.49", "0.50"),
    ]
    replacement = [
        _quote("entry", 0, "0.49", "0.70"),
        _quote("horizon", 60, "0.20", "0.21"),
    ]
    parameters = _parameters()
    hashes = compute_x05_runtime_hashes(first, [T0], parameters)
    lock_ids = (
        "x05_quote_manifest",
        "barrier_values",
        "purge_and_embargo",
        "post_resume_quote_rule",
        "same_time_touch_rule",
    )
    locks = [
        {
            "id": lock_id,
            "status": "resolved",
            "evidence_ref": hashes.data_sha256
            if lock_id == "x05_quote_manifest"
            else hashes.configuration_sha256,
        }
        for lock_id in lock_ids
    ]
    fake_registry = {
        "X-05": {
            "execution_authorized": True,
            "authorization_scopes": {
                "label_generation": {
                    "authorized": True,
                    "required_lock_ids": list(lock_ids),
                }
            },
            "registration_locks": locks,
            "preregistered_inputs": {
                "label_generation": {
                    "code_sha256": hashes.code_sha256,
                    "data_sha256": hashes.data_sha256,
                }
            },
            "dependencies": [],
        }
    }
    monkeypatch.setattr(labels_module, "load_experiment_registry", lambda _: fake_registry)

    class SwappingSequence:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(first if self.iterations == 1 else replacement)

    source = SwappingSequence()
    labels = generate_x05_long_labels(
        PROJECT_ROOT,
        quotes=source,  # type: ignore[arg-type]
        anchors=[T0],
        parameters=parameters,
    )

    assert source.iterations == 1
    assert labels[0].entry_price == _price("0.50")


def test_anchor_and_quote_times_must_be_utc() -> None:
    naive = datetime(2026, 7, 22, 18, 0)
    with pytest.raises(LabelInputError, match="UTC"):
        label_long(
            [_quote("entry", 0, "0.49", "0.50")],
            anchor_at=naive,
            parameters=_parameters(),
        )
