from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from prediction_market import contracts


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _digest(fill: str) -> str:
    return "sha256:" + fill * 64


def _step() -> dict[str, object]:
    value: dict[str, object] = {
        "step_version": "v0",
        "sport": "nfl",
        "game_id": "game_nfl_2025_01_DAL_PHI",
        "sequence": 18,
        "terminal": False,
        "reducer_id": "REDUCER-NFL-PLAY",
        "reducer_version": "v1",
        "state_schema_id": "urn:saf:game-state:nfl:v1",
        "event_schema_id": "urn:saf:game-event:nfl:v1",
        "event_id": "evt_" + "a" * 64,
        "previous_state_sha256": _digest("1"),
        "event_sha256": _digest("2"),
        "next_state_sha256": _digest("3"),
        "observation_mode": "offline_reconstruction",
        "quality_flags": ["source_clock_unverified"],
        "step_sha256": _digest("0"),
    }
    value["step_sha256"] = contracts.game_state_step_sha256(value)
    return value


def test_game_state_step_contract_is_immutable_and_self_hashed() -> None:
    validated = contracts.validate_contract_v0(
        "game-state-step/v0.schema.yaml",
        _step(),
    )

    assert validated.sport == "nfl"
    assert validated.sequence == 18
    assert validated.quality_flags == ("source_clock_unverified",)
    with pytest.raises(ValidationError):
        validated.sequence = 19


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sport", "cricket", "sport"),
        ("game_id", "not-canonical", "game_id"),
        ("sequence", -1, "sequence"),
        ("event_id", "evt_bad", "event_id"),
        ("previous_state_sha256", "sha256:bad", "previous_state_sha256"),
        ("observation_mode", "unknown", "observation_mode"),
    ],
)
def test_game_state_step_rejects_invalid_identity(
    field: str,
    value: object,
    message: str,
) -> None:
    document = _step()
    document[field] = value
    document["step_sha256"] = contracts.game_state_step_sha256(document)
    with pytest.raises(ValidationError, match=message):
        contracts.validate_contract_v0(
            "game-state-step/v0.schema.yaml",
            document,
        )


def test_game_state_step_rejects_tampering_and_duplicate_flags() -> None:
    tampered = copy.deepcopy(_step())
    tampered["terminal"] = True
    with pytest.raises(ValidationError, match="step_sha256"):
        contracts.validate_contract_v0(
            "game-state-step/v0.schema.yaml",
            tampered,
        )

    duplicate = _step()
    duplicate["quality_flags"] = [
        "source_clock_unverified",
        "source_clock_unverified",
    ]
    duplicate["step_sha256"] = contracts.game_state_step_sha256(duplicate)
    with pytest.raises(ValidationError, match="quality_flags"):
        contracts.validate_contract_v0(
            "game-state-step/v0.schema.yaml",
            duplicate,
        )


def test_schema_declares_normative_semantic_validator() -> None:
    schema = (
        PROJECT_ROOT / "contracts" / "game-state-step" / "v0.schema.yaml"
    ).read_text(encoding="utf-8")
    assert "prediction_market.contracts.validate_contract_v0" in schema
    assert "additionalProperties: false" in schema
