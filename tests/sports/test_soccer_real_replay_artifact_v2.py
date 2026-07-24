from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from prediction_market.contracts import canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "game-state"
    / "soccer_real_replay_validation_v2.json"
)


def _artifact() -> dict[str, Any]:
    document = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _require_valid_self_hash(document: dict[str, Any]) -> None:
    material = deepcopy(document)
    claimed = material.pop("artifact_sha256")
    if claimed != canonical_sha256(material):
        raise ValueError("soccer replay artifact self-hash mismatch")


def test_soccer_replay_artifact_has_a_valid_canonical_self_hash() -> None:
    _require_valid_self_hash(_artifact())


@pytest.mark.parametrize("field", ["generated_at", "events_reduced"])
def test_soccer_replay_artifact_self_hash_detects_tampering(field: str) -> None:
    document = _artifact()
    run_hash = document["season_replay"]["canonical_state_sha256"]
    tampered = deepcopy(document)
    if field == "generated_at":
        tampered["generated_at"] = "2026-07-24T00:00:00Z"
    else:
        tampered["season_replay"]["per_run"]["events_reduced"] += 1

    assert tampered["season_replay"]["canonical_state_sha256"] == run_hash
    with pytest.raises(ValueError, match="self-hash mismatch"):
        _require_valid_self_hash(tampered)
