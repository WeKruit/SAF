"""Quality flags and counters used by PMXT reconstruction.

PMXT v2 has no exchange sequence number.  Consequently a long receive-time
interval is evidence of a gap candidate only; it is not proof of packet loss.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


CROSSED_BOOK = "CROSSED_BOOK"
DUPLICATE_EVENT = "DUPLICATE_EVENT"
MISSING_INITIAL_SNAPSHOT = "MISSING_INITIAL_SNAPSHOT"
NONPOSITIVE_SIZE = "NONPOSITIVE_SIZE"
OUT_OF_ORDER = "OUT_OF_ORDER"
RECEIVE_GAP_CANDIDATE = "RECEIVE_GAP_CANDIDATE"


@dataclass
class QualityTracker:
    """Accumulate deterministic flags and integer observations."""

    flags: set[str] = field(default_factory=set)
    counts: Counter[str] = field(default_factory=Counter)

    def mark(self, flag: str, counter: str, amount: int = 1) -> None:
        self.flags.add(flag)
        self.counts[counter] += amount

    def sorted_flags(self) -> tuple[str, ...]:
        return tuple(sorted(self.flags))
