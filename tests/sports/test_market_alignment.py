from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from prediction_market.sports.market_alignment import (
    CanonicalGameConditionBinding,
    ExecutableQuoteObservation,
    MarketAlignmentInputError,
    ModelProbabilityObservation,
    evaluate_market_alignment,
)


UTC = timezone.utc
SHA_A = "sha256:" + ("a" * 64)
SHA_B = "sha256:" + ("b" * 64)
SHA_C = "sha256:" + ("c" * 64)


def _at(second: int, millisecond: int = 0) -> datetime:
    return datetime(2026, 7, 23, 12, 0, second, millisecond * 1_000, tzinfo=UTC)


def _binding() -> CanonicalGameConditionBinding:
    return CanonicalGameConditionBinding(
        game_id="game_nfl_2026_example",
        condition_id="condition_polymarket_example",
        outcome_id="outcome_home_win",
        metadata_snapshot_ref=SHA_A,
        metadata_observed_at=_at(1),
    )


def _model() -> ModelProbabilityObservation:
    return ModelProbabilityObservation(
        game_id="game_nfl_2026_example",
        outcome_id="outcome_home_win",
        cutoff_at=_at(10),
        probability=Decimal("0.61"),
        model_output_ref=SHA_B,
    )


def _quote() -> ExecutableQuoteObservation:
    return ExecutableQuoteObservation(
        game_id="game_nfl_2026_example",
        condition_id="condition_polymarket_example",
        outcome_id="outcome_home_win",
        received_at=_at(9, 500),
        best_bid=Decimal("0.55"),
        best_ask=Decimal("0.58"),
        bid_depth=Decimal("125"),
        ask_depth=Decimal("80"),
        midpoint=None,
        paused=False,
        metadata_snapshot_ref=SHA_A,
        rule_snapshot_ref=SHA_C,
        rule_snapshot_observed_at=_at(2),
    )


def _evaluate(
    *,
    binding: CanonicalGameConditionBinding | None = None,
    model: ModelProbabilityObservation | None = None,
    quote: ExecutableQuoteObservation | None = None,
):
    return evaluate_market_alignment(
        binding=binding or _binding(),
        model=model or _model(),
        quote=quote or _quote(),
        max_quote_age_ms=1_000,
    )


def test_ready_input_outside_executable_interval_is_only_predictive_disagreement() -> None:
    decision = _evaluate()

    assert decision.status == "predictive_disagreement"
    assert decision.reason_codes == ()
    assert decision.as_of_age_ms == 500
    assert decision.spread == Decimal("0.03")
    assert decision.probability_distance_to_executable_interval == Decimal("0.03")
    assert set(decision.to_dict()) == {
        "status",
        "reason_codes",
        "comparison_basis",
        "as_of_age_ms",
        "spread",
        "probability_distance_to_executable_interval",
    }
    assert "alpha" not in str(decision.to_dict()).lower()


def test_probability_inside_executable_interval_is_alignment_ready() -> None:
    decision = _evaluate(
        model=replace(_model(), probability=Decimal("0.56"))
    )

    assert decision.status == "alignment_ready"
    assert decision.reason_codes == ()
    assert decision.probability_distance_to_executable_interval == Decimal("0")


def test_future_quote_is_rejected() -> None:
    decision = _evaluate(quote=replace(_quote(), received_at=_at(10, 1)))

    assert decision.status == "not_aligned"
    assert decision.reason_codes == ("future_quote",)


def test_midpoint_only_quote_is_rejected() -> None:
    decision = _evaluate(
        quote=replace(
            _quote(),
            best_bid=None,
            best_ask=None,
            bid_depth=None,
            ask_depth=None,
            midpoint=Decimal("0.565"),
        )
    )

    assert decision.status == "not_aligned"
    assert decision.reason_codes == ("midpoint_only_forbidden",)


@pytest.mark.parametrize(
    ("quote", "reason"),
    [
        (
            replace(_quote(), best_bid=None, bid_depth=None),
            "missing_executable_bid",
        ),
        (
            replace(_quote(), best_ask=None, ask_depth=None),
            "missing_executable_ask",
        ),
    ],
)
def test_missing_bid_or_ask_is_rejected(
    quote: ExecutableQuoteObservation,
    reason: str,
) -> None:
    decision = _evaluate(quote=quote)

    assert decision.status == "not_aligned"
    assert reason in decision.reason_codes


def test_stale_join_is_rejected() -> None:
    decision = _evaluate(quote=replace(_quote(), received_at=_at(8, 999)))

    assert decision.status == "not_aligned"
    assert decision.reason_codes == ("stale_quote",)
    assert decision.as_of_age_ms == 1_001


@pytest.mark.parametrize(
    ("model", "quote", "reason"),
    [
        (
            replace(_model(), game_id="game_nfl_wrong"),
            _quote(),
            "model_game_mismatch",
        ),
        (
            _model(),
            replace(_quote(), game_id="game_nfl_wrong"),
            "quote_game_mismatch",
        ),
        (
            _model(),
            replace(_quote(), condition_id="condition_polymarket_wrong"),
            "quote_condition_mismatch",
        ),
        (
            replace(_model(), outcome_id="outcome_away_win"),
            _quote(),
            "model_outcome_mismatch",
        ),
    ],
)
def test_wrong_game_condition_or_outcome_is_rejected(
    model: ModelProbabilityObservation,
    quote: ExecutableQuoteObservation,
    reason: str,
) -> None:
    decision = _evaluate(model=model, quote=quote)

    assert decision.status == "not_aligned"
    assert reason in decision.reason_codes


@pytest.mark.parametrize(
    ("quote", "reason"),
    [
        (replace(_quote(), paused=True), "market_paused"),
        (replace(_quote(), bid_depth=Decimal("0")), "non_executable_bid_depth"),
        (replace(_quote(), ask_depth=Decimal("0")), "non_executable_ask_depth"),
        (
            replace(_quote(), rule_snapshot_ref=None),
            "missing_rule_snapshot",
        ),
        (
            replace(_quote(), rule_snapshot_observed_at=_at(10, 1)),
            "future_rule_snapshot",
        ),
        (
            replace(_quote(), metadata_snapshot_ref=SHA_B),
            "metadata_snapshot_mismatch",
        ),
    ],
)
def test_pause_depth_and_point_in_time_evidence_are_fail_closed(
    quote: ExecutableQuoteObservation,
    reason: str,
) -> None:
    decision = _evaluate(quote=quote)

    assert decision.status == "not_aligned"
    assert reason in decision.reason_codes


def test_future_metadata_binding_is_rejected() -> None:
    decision = _evaluate(
        binding=replace(_binding(), metadata_observed_at=_at(10, 1))
    )

    assert decision.status == "not_aligned"
    assert decision.reason_codes == ("future_metadata_snapshot",)


def test_binary_float_and_non_utc_timestamp_are_rejected_at_input_boundary() -> None:
    with pytest.raises(MarketAlignmentInputError, match="binary float"):
        replace(_model(), probability=0.61)  # type: ignore[arg-type]

    with pytest.raises(MarketAlignmentInputError, match="UTC"):
        replace(_quote(), received_at=datetime(2026, 7, 23, 12, 0, 9))
