from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "pmxt"

sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def pmxt_fixture() -> Path:
    return FIXTURE_ROOT / "l2_events.jsonl"


@pytest.fixture
def anomalous_fixture() -> list[dict[str, str]]:
    return [
        {
            "timestamp_received": "2026-06-01T00:00:00.100Z",
            "timestamp": "2026-06-01T00:00:00.090Z",
            "market": "0xanomalous",
            "event_type": "price_change",
            "asset_id": "asset-x",
            "price": "0.5000",
            "size": "-1.000000",
            "side": "BUY",
        },
        {
            "timestamp_received": "2026-06-01T00:00:00.200Z",
            "timestamp": "2026-06-01T00:00:00.190Z",
            "market": "0xanomalous",
            "event_type": "book",
            "asset_id": "asset-x",
            "bids": ('[["0.6000","10.000000"],["0.5000","0.000000"]]'),
            "asks": '[["0.5500","8.000000"]]',
        },
    ]
