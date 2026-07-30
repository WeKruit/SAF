from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from prediction_market.research import nfl_x15_models as models


_LEGACY_SMALL_RUN_DIGESTS = {
    "oof_predictions": (
        "1f6784286337e5678a95bcdcb0eebeedc7fa8f35b1698d06f76f53f88f37f100"
    ),
    "conditional_quantiles": (
        "7b13a3b160e92d7fcce0588b0264b75c4b874056e89598a35ee17aff4e1cac45"
    ),
    "fold_metrics": (
        "cdc45749ae9c9595ae28dc248b25e9b9c7677f8ee324609a4f9f10f4dc71786c"
    ),
    "support_audit": (
        "89b511a8a27ec23c6e4901c677d9814acc7e97a9fa164970f7fe8326fd3c5745"
    ),
    "weight_audit": (
        "9604d5a7b5d5f22ef5514fe8bb3900681a1e74f4cfc51ac013dcfc1d94a20499"
    ),
}
_LEGACY_RUN_CONFIG_SHA256 = (
    "sha256:891965c35b7a4bfba48d9835343420c451ef89b84c1c2cda45495c555f1da231"
)


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    return json.dumps(
        {
            "columns": list(frame.columns),
            "dtypes": [str(value) for value in frame.dtypes],
            "records": frame.to_dict(orient="records"),
        },
        default=str,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")


def _legacy_training_data_hash(frame: pd.DataFrame) -> str:
    rows = sorted(
        [
            {
                "game_id": str(row["game_id"]),
                "episode_id": str(row["atomic_information_episode_id"]),
                "source_row_id": int(row["_source_row_id"]),
                "venue": str(row["venue"]),
                "decision_feature_sha256": str(
                    row["decision_feature_sha256"]
                ),
                "s_h": models._canonical(row["s_h"]),
                "o_h_given_s": models._canonical(row["o_h_given_s"]),
                "direction": models._canonical(row["_direction_truth"]),
                "magnitude": models._canonical(
                    row["_conditional_magnitude"]
                ),
            }
            for _, row in frame.iterrows()
        ],
        key=lambda item: (
            item["game_id"],
            item["episode_id"],
            item["source_row_id"],
            item["venue"],
            item["decision_feature_sha256"],
            json.dumps(item, sort_keys=True),
        ),
    )
    return models._sha256(rows)


def _legacy_training_hash(
    frame: pd.DataFrame,
    *,
    model_id: str,
    block_id: str,
    head: str,
    weights: pd.Series,
) -> str:
    mask = models._head_mask(frame, head)
    rows = sorted(
        [
            {
                "game_id": str(row.game_id),
                "episode_id": str(row.atomic_information_episode_id),
                "source_row_id": int(row._source_row_id),
                "decision_feature_sha256": str(
                    row.decision_feature_sha256
                ),
                "truth": models._canonical(
                    row._s_interval_truth
                    if head == "S_H"
                    else row.o_h_given_s
                    if head == "O_H_GIVEN_S"
                    else row._direction_truth
                ),
                "weight": float(weights.loc[index]),
            }
            for index, row in frame.loc[mask].iterrows()
        ],
        key=lambda item: (
            item["game_id"],
            item["episode_id"],
            item["source_row_id"],
            item["decision_feature_sha256"],
            json.dumps(item["truth"], sort_keys=True),
            item["weight"],
        ),
    )
    return models._sha256(
        {
            "model_id": model_id,
            "feature_block_id": block_id,
            "head": head,
            "rows": rows,
        }
    )


def _hash_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g2", "g1", "g1", "g1", "g2", "g2"],
            "atomic_information_episode_id": [
                "e2",
                "e1",
                "e1",
                "e1",
                "e2",
                "e3",
            ],
            "_source_row_id": [5, 2, 2, 2, 4, 6],
            "venue": [
                "kalshi",
                "polymarket",
                "polymarket",
                "polymarket",
                "kalshi",
                "kalshi",
            ],
            "decision_feature_sha256": [
                "sha256:" + character * 64
                for character in ("f", "a", "a", "a", "e", "d")
            ],
            "s_h": [True, True, False, pd.NA, False, True],
            "o_h_given_s": [True, False, pd.NA, pd.NA, pd.NA, True],
            "_direction_truth": [
                "UP",
                "DOWN",
                pd.NA,
                pd.NA,
                pd.NA,
                "NO_MOVE",
            ],
            "_conditional_magnitude": [
                0.02,
                0.01,
                np.nan,
                np.nan,
                np.nan,
                0.0,
            ],
            "_s_interval_at_risk": [True, True, True, False, True, True],
            "_s_interval_truth": [True, True, False, pd.NA, False, True],
            "target_eligible": [True, True, False, False, False, True],
        }
    )


def test_streaming_json_hash_is_byte_identical_to_canonical_hash() -> None:
    rows = [
        {"z": 1.5, "a": None, "text": "中"},
        {"z": -0.0, "a": True, "text": '"quoted"'},
    ]

    assert models._sha256_row_stream(iter(rows)) == models._sha256(rows)
    assert models._sha256_row_stream(
        iter(rows),
        envelope={"model_id": "m", "head": ("UP", "DOWN")},
    ) == models._sha256(
        {
            "model_id": "m",
            "head": ("UP", "DOWN"),
            "rows": rows,
        }
    )


def test_streamed_training_hashes_match_legacy_with_tied_identity_rows() -> None:
    frame = _hash_fixture()

    assert models._training_data_hash(frame) == _legacy_training_data_hash(
        frame
    )
    for head in ("S_H", "O_H_GIVEN_S", "DIRECTION"):
        mask = models._head_mask(frame, head)
        weights = models.hierarchical_sample_weights(frame, mask)
        assert models._training_hash(
            frame,
            model_id="regularized_logistic_v1",
            block_id="D3",
            head=head,
            weights=weights,
        ) == _legacy_training_hash(
            frame,
            model_id="regularized_logistic_v1",
            block_id="D3",
            head=head,
            weights=weights,
        )


def test_small_oof_outputs_and_run_config_match_preoptimization_bytes() -> None:
    namespace = runpy.run_path(
        str(
            Path(__file__).parent
            / "research"
            / "test_nfl_x15_models.py"
        )
    )
    result = namespace["_small_run"](namespace["_panel"]())

    for name, expected in _LEGACY_SMALL_RUN_DIGESTS.items():
        actual = hashlib.sha256(
            _frame_bytes(getattr(result, name))
        ).hexdigest()
        assert actual == expected
    assert result.run_config_sha256 == _LEGACY_RUN_CONFIG_SHA256


def test_model_hotpath_contains_no_iterrows() -> None:
    source = Path(models.__file__).read_text(encoding="utf-8")
    assert ".iterrows(" not in source
