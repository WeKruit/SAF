from __future__ import annotations

import pandas as pd
import pytest

from prediction_market.research.nfl_x16_kalshi_validation import (
    KalshiValidationError,
    append_holdout_access,
    begin_holdout_access_ledger,
    fit_polymarket_source_calibrator,
    lock_holdout_validation,
    transport_to_kalshi_holdout,
)


AUTHORITY = "sha256:" + "b" * 64


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["dev-1", "dev-2", "dev-3", "dev-4"],
            "nfl_week": [1, 2, 3, 4],
            "cohort": ["development"] * 4,
            "authority_sha256": [AUTHORITY] * 4,
            "venue": ["polymarket"] * 4,
            "raw_probability": [0.15, 0.30, 0.70, 0.85],
            "outcome": [0, 0, 1, 1],
        }
    )


def _target() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["holdout-1", "holdout-2"],
            "nfl_week": [13, 18],
            "cohort": ["holdout", "holdout"],
            "authority_sha256": [AUTHORITY, AUTHORITY],
            "venue": ["kalshi", "kalshi"],
            "raw_probability": [0.25, 0.75],
        }
    )


def test_transport_fits_and_calibrates_only_polymarket_development() -> None:
    calibrator = fit_polymarket_source_calibrator(_source(), authority_sha256=AUTHORITY)
    transported = transport_to_kalshi_holdout(
        _target(), calibrator=calibrator, authority_sha256=AUTHORITY
    )

    assert transported["transport_source"].eq("polymarket_development_only").all()
    assert transported["calibrated_probability"].between(0, 1).all()

    with_outcome = _target().assign(outcome=[0, 1])
    with pytest.raises(KalshiValidationError, match="outcome"):
        transport_to_kalshi_holdout(
            with_outcome, calibrator=calibrator, authority_sha256=AUTHORITY
        )


def test_dynamic_holdout_access_chain_locks_only_before_any_reaction_read() -> None:
    ledger = begin_holdout_access_ledger(authority_sha256=AUTHORITY)
    locked = lock_holdout_validation(ledger, authority_sha256=AUTHORITY)
    assert locked["holdout_reaction_read_count"] == 0
    assert locked["chain_sha256"].startswith("sha256:")

    opened = append_holdout_access(ledger, access_kind="holdout_reaction_read")
    with pytest.raises(KalshiValidationError, match="read count=0"):
        lock_holdout_validation(opened, authority_sha256=AUTHORITY)
