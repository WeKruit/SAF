"""Quality flags and counters used by PMXT reconstruction.

PMXT v2 has no exchange sequence number.  Consequently a long receive-time
interval is evidence of a gap candidate only; it is not proof of packet loss.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from prediction_market.contracts import validate_contract_v0


CROSSED_BOOK = "crossed_book"
DUPLICATE_EVENT = "duplicate_event"
MISSING_INITIAL_SNAPSHOT = "missing_initial_snapshot"
NONPOSITIVE_SIZE = "non_positive_size"
OUT_OF_ORDER = "out_of_order"


@dataclass
class QualityTracker:
    """Accumulate deterministic flags and integer observations."""

    flags: set[str] = field(default_factory=set)
    counts: Counter[str] = field(default_factory=Counter)

    def mark(self, flag: str, counter: str, amount: int = 1) -> None:
        validated = validate_contract_v0("quality-flags/v0.yaml", flag)
        self.flags.add(validated)
        self.counts[counter] += amount

    def sorted_flags(self) -> tuple[str, ...]:
        return tuple(sorted(self.flags))
