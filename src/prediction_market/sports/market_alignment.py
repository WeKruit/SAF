"""Fail-closed audit of current sport/market alignment prerequisites.

The program has no registered game-state-to-market outcome binding and no
registered maximum quote-age policy.  Consequently this v0 module can verify
content-addressed prerequisite documents, but it cannot emit a comparison,
disagreement, profitability, or alpha result.  A future comparison interface
requires those governed inputs rather than caller-selected hashes or policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from prediction_market.contracts import (
    EventEnvelopeV0,
    FixedPointV0,
    MarketMetadataSnapshotV0,
    ModelOutputV1,
    VenueRuleSnapshotV0,
    thaw_contract_v0,
    validate_event_envelope_v0,
)


_ZERO = Decimal("0")
_ONE = Decimal("1")
_QUOTE_PAYLOAD_FIELDS = frozenset(
    {
        "kind",
        "best_bid",
        "best_ask",
        "bid_depth",
        "ask_depth",
        "paused",
    }
)


class MarketAlignmentInputError(ValueError):
    """Raised when claimed evidence is not a validated governed document."""


def _require_tuple_of(
    value: object,
    expected_type: type[object],
    field_name: str,
) -> None:
    if type(value) is not tuple:
        raise MarketAlignmentInputError(f"{field_name} must be a tuple")
    if any(not isinstance(item, expected_type) for item in value):
        raise MarketAlignmentInputError(
            f"{field_name} must contain validated {expected_type.__name__} documents"
        )


@dataclass(frozen=True, slots=True)
class CurrentAlignmentEvidence:
    """The governed documents currently available to the prerequisite audit."""

    metadata_snapshots: tuple[MarketMetadataSnapshotV0, ...] = ()
    model_output_events: tuple[EventEnvelopeV0, ...] = ()
    quote_events: tuple[EventEnvelopeV0, ...] = ()
    rule_snapshots: tuple[VenueRuleSnapshotV0, ...] = ()

    def __post_init__(self) -> None:
        _require_tuple_of(
            self.metadata_snapshots,
            MarketMetadataSnapshotV0,
            "metadata_snapshots",
        )
        _require_tuple_of(
            self.model_output_events,
            EventEnvelopeV0,
            "model_output_events",
        )
        _require_tuple_of(
            self.quote_events,
            EventEnvelopeV0,
            "quote_events",
        )
        _require_tuple_of(
            self.rule_snapshots,
            VenueRuleSnapshotV0,
            "rule_snapshots",
        )


@dataclass(frozen=True, slots=True)
class CurrentAlignmentDecision:
    """A non-promotional statement about verified prerequisite availability."""

    status: Literal["not_aligned"]
    reason_codes: tuple[str, ...]
    verified_document_counts: Mapping[str, int]
    matched_as_of_rows: Literal[0] = 0
    comparison_basis: Literal[
        "verified_prerequisites_only"
    ] = "verified_prerequisites_only"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "verified_document_counts": dict(self.verified_document_counts),
            "matched_as_of_rows": self.matched_as_of_rows,
            "comparison_basis": self.comparison_basis,
        }


def _utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except (TypeError, ValueError) as exc:
        raise MarketAlignmentInputError("contract timestamp is not valid UTC") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise MarketAlignmentInputError("contract timestamp is not valid UTC")
    return parsed


def _revalidate_metadata(
    snapshot: MarketMetadataSnapshotV0,
) -> MarketMetadataSnapshotV0:
    try:
        return MarketMetadataSnapshotV0.model_validate(
            snapshot.model_dump(mode="python", round_trip=True)
        )
    except (TypeError, ValueError) as exc:
        raise MarketAlignmentInputError(
            "invalid MarketMetadataSnapshotV0 evidence"
        ) from exc


def _revalidate_rule(snapshot: VenueRuleSnapshotV0) -> VenueRuleSnapshotV0:
    try:
        return VenueRuleSnapshotV0.model_validate(
            snapshot.model_dump(mode="python", round_trip=True)
        )
    except (TypeError, ValueError) as exc:
        raise MarketAlignmentInputError(
            "invalid VenueRuleSnapshotV0 evidence"
        ) from exc


def _validated_model_event(
    program_root: Path,
    event: EventEnvelopeV0,
) -> tuple[EventEnvelopeV0, ModelOutputV1]:
    try:
        validated = validate_event_envelope_v0(program_root, event)
    except (TypeError, ValueError) as exc:
        raise MarketAlignmentInputError(
            "invalid registry-backed model-output EventEnvelopeV0"
        ) from exc
    if validated.event_type != "model_output":
        raise MarketAlignmentInputError(
            "model_output_events must be model_output EventEnvelopeV0 documents"
        )
    try:
        output = ModelOutputV1.model_validate(thaw_contract_v0(validated.payload))
    except (TypeError, ValueError) as exc:
        raise MarketAlignmentInputError("invalid ModelOutputV1 payload") from exc
    if validated.canonical_refs.game_id != output.game_id:
        raise MarketAlignmentInputError(
            "model-output envelope game_id does not match its payload"
        )
    if _utc(validated.time.receive_at) < _utc(output.pit_cutoff_at):
        raise MarketAlignmentInputError(
            "model-output availability precedes its PIT cutoff"
        )
    return validated, output


def _fixed_point(
    value: object,
    field_name: str,
    *,
    probability: bool,
) -> Decimal:
    try:
        fixed = FixedPointV0.model_validate(value)
    except (TypeError, ValueError) as exc:
        raise MarketAlignmentInputError(
            f"quote {field_name} must be exact FixedPointV0"
        ) from exc
    decimal_value = fixed.to_decimal()
    if probability:
        if decimal_value < _ZERO or decimal_value > _ONE:
            raise MarketAlignmentInputError(
                f"quote {field_name} must be in [0, 1]"
            )
    elif decimal_value <= _ZERO:
        raise MarketAlignmentInputError(
            f"quote {field_name} must be positive"
        )
    return decimal_value


def _validated_quote_event(
    program_root: Path,
    event: EventEnvelopeV0,
) -> EventEnvelopeV0:
    try:
        validated = validate_event_envelope_v0(program_root, event)
    except (TypeError, ValueError) as exc:
        raise MarketAlignmentInputError(
            "invalid quote EventEnvelopeV0 evidence"
        ) from exc
    if validated.event_type != "normalized_observation":
        raise MarketAlignmentInputError(
            "quote evidence must be a normalized_observation"
        )
    if validated.time.receive_basis != "local_recorder":
        raise MarketAlignmentInputError(
            "quote evidence requires local_recorder receive time"
        )
    if validated.source.venue is None:
        raise MarketAlignmentInputError("quote evidence requires an observed venue")
    canonical = validated.canonical_refs
    if (
        canonical.game_id is None
        or canonical.condition_id is None
        or canonical.outcome_id is None
    ):
        raise MarketAlignmentInputError(
            "quote evidence requires canonical game/condition/outcome"
        )
    payload = thaw_contract_v0(validated.payload)
    if set(payload) != _QUOTE_PAYLOAD_FIELDS:
        raise MarketAlignmentInputError(
            "quote payload must contain the exact executable-quote fields"
        )
    if payload["kind"] != "executable_quote":
        raise MarketAlignmentInputError(
            "quote payload kind must be executable_quote"
        )
    if type(payload["paused"]) is not bool or payload["paused"]:
        raise MarketAlignmentInputError(
            "quote evidence must explicitly observe paused=false"
        )
    bid = _fixed_point(payload["best_bid"], "best_bid", probability=True)
    ask = _fixed_point(payload["best_ask"], "best_ask", probability=True)
    _fixed_point(payload["bid_depth"], "bid_depth", probability=False)
    _fixed_point(payload["ask_depth"], "ask_depth", probability=False)
    if bid > ask:
        raise MarketAlignmentInputError("quote evidence contains a crossed book")
    return validated


def _validate_mutual_identity(
    metadata: tuple[MarketMetadataSnapshotV0, ...],
    models: tuple[tuple[EventEnvelopeV0, ModelOutputV1], ...],
    quotes: tuple[EventEnvelopeV0, ...],
    rules: tuple[VenueRuleSnapshotV0, ...],
) -> None:
    if not (len(metadata) == len(models) == len(quotes) == len(rules) == 1):
        return
    metadata_item = metadata[0]
    model_event, model_output = models[0]
    quote = quotes[0]
    rule = rules[0]
    canonical = metadata_item.canonical_refs
    if (
        quote.canonical_refs.game_id != canonical.game_id
        or quote.canonical_refs.condition_id != canonical.condition_id
        or quote.canonical_refs.outcome_id != canonical.outcome_id
        or model_event.canonical_refs.game_id != canonical.game_id
        or model_output.game_id != canonical.game_id
    ):
        raise MarketAlignmentInputError(
            "metadata/model/quote canonical identity mismatch"
        )
    if (
        rule.venue != metadata_item.venue
        or rule.venue != quote.source.venue
        or rule.condition_id != canonical.condition_id
    ):
        raise MarketAlignmentInputError(
            "metadata/quote/rule venue or condition mismatch"
        )
    cutoff = _utc(model_output.pit_cutoff_at)
    if _utc(metadata_item.captured_at) > cutoff:
        raise MarketAlignmentInputError(
            "metadata snapshot is later than the model cutoff"
        )
    if _utc(quote.time.receive_at) > cutoff:
        raise MarketAlignmentInputError(
            "quote observation is later than the model cutoff"
        )
    if _utc(rule.fetched_at) > cutoff or _utc(rule.effective_from) > cutoff:
        raise MarketAlignmentInputError(
            "venue-rule snapshot is later than the model cutoff"
        )


def audit_current_market_alignment_evidence(
    *,
    program_root: str | Path,
    evidence: CurrentAlignmentEvidence,
) -> CurrentAlignmentDecision:
    """Verify current documents without inventing missing binding or age policy."""

    if not isinstance(evidence, CurrentAlignmentEvidence):
        raise MarketAlignmentInputError(
            "evidence must be CurrentAlignmentEvidence"
        )
    root = Path(program_root)
    metadata = tuple(
        _revalidate_metadata(snapshot)
        for snapshot in evidence.metadata_snapshots
    )
    models = tuple(
        _validated_model_event(root, event)
        for event in evidence.model_output_events
    )
    quotes = tuple(
        _validated_quote_event(root, event)
        for event in evidence.quote_events
    )
    rules = tuple(
        _revalidate_rule(snapshot)
        for snapshot in evidence.rule_snapshots
    )
    _validate_mutual_identity(metadata, models, quotes, rules)

    reasons = ["missing_canonical_game_condition_outcome_binding"]
    if not metadata:
        reasons.append("missing_market_metadata_snapshot")
    if not models:
        reasons.append("missing_model_output")
    if not quotes:
        reasons.append("missing_local_receive_executable_quote")
    if not rules:
        reasons.append("missing_venue_rule_snapshot")
    reasons.append("missing_registered_join_policy")
    return CurrentAlignmentDecision(
        status="not_aligned",
        reason_codes=tuple(reasons),
        verified_document_counts=MappingProxyType(
            {
                "metadata_snapshots": len(metadata),
                "model_output_events": len(models),
                "quote_events": len(quotes),
                "rule_snapshots": len(rules),
            }
        ),
    )


__all__ = [
    "CurrentAlignmentDecision",
    "CurrentAlignmentEvidence",
    "MarketAlignmentInputError",
    "audit_current_market_alignment_evidence",
]
