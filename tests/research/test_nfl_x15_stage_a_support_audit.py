from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from prediction_market.research.nfl_x15_development_panel import (
    VerifiedDiagnosticPanelPartition,
)
from prediction_market.research.nfl_x15_stage_a_support_audit import (
    StageASupportAuditError,
    _summarize_partitions,
)


_SHA_A = "sha256:" + "a" * 64
_SHA_B = "sha256:" + "b" * 64
_SHA_C = "sha256:" + "c" * 64
_SHA_D = "sha256:" + "d" * 64
_SHA_E = "sha256:" + "e" * 64


def _partition(game_number: int) -> VerifiedDiagnosticPanelPartition:
    frame = pd.DataFrame(
        {
            "decision_eligible": [True, True, False],
            "stage_a_status": [
                "NOT_KNOWN_AT_L",
                "UNAVAILABLE",
                "AVAILABLE",
            ],
            "p_before_home": [pd.NA, pd.NA, 0.4],
            "p_after_home": [pd.NA, pd.NA, 0.5],
            "reference_delta_home": [pd.NA, pd.NA, 0.1],
            "reference_gap_at_landmark": [pd.NA, pd.NA, 0.02],
        }
    )
    return VerifiedDiagnosticPanelPartition(
        game_id=f"game-{game_number:03d}",
        panel=frame,
        batch_game_count=153,
        batch_manifest_file_sha256=_SHA_A,
        batch_sha256=_SHA_B,
        game_manifest_sha256=_SHA_E,
        diagnostic_object_sha256=_SHA_E,
        cohort_authority_sha256=_SHA_C,
        cohort_mapping_sha256=_SHA_D,
    )


def test_zero_decision_reference_support_blocks_d4() -> None:
    summary = _summarize_partitions(
        _partition(index) for index in range(153)
    )

    assert summary["decision_eligible_rows"] == 306
    assert summary["total_panel_rows"] == 459
    assert summary["nonnull_counts"] == {
        "p_before_home": 0,
        "p_after_home": 0,
        "reference_delta_home": 0,
        "reference_gap_at_landmark": 0,
    }
    assert summary["feature_block_status"] == {
        "D4": "UNSUPPORTED_FEATURE_BLOCK"
    }
    assert summary["selection_eligible"] == {"D4": False}
    assert summary["overall_nonnull_counts"]["p_before_home"] == 153


def test_duplicate_game_partition_fails_closed() -> None:
    partitions = [_partition(index) for index in range(152)]
    partitions.append(_partition(0))

    with pytest.raises(
        StageASupportAuditError, match="duplicate verified game"
    ):
        _summarize_partitions(partitions)


def test_non_boolean_decision_flag_fails_closed() -> None:
    partition = _partition(0)
    bad = replace(
        partition,
        panel=partition.panel.assign(decision_eligible=[1, 0, 1]),
    )
    partitions = [bad, *(_partition(index) for index in range(1, 153))]

    with pytest.raises(
        StageASupportAuditError, match="decision_eligible is not boolean"
    ):
        _summarize_partitions(partitions)
