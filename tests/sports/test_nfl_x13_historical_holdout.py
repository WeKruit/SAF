from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prediction_market.sports import nfl_x13_historical_holdout as holdout
from prediction_market.sports.nfl_x13_holdout_evidence import (
    HoldoutEvidenceError,
)


def _verified_index(count: int = 24) -> dict[str, object]:
    games: list[dict[str, object]] = []
    for index in range(count):
        game_id = f"2025_{index + 1:02d}_AAA_BBB"
        games.append(
            {
                "game_id": game_id,
                "schedule_evidence": {
                    "manifest_sha256": "sha256:" + "a" * 64,
                },
                "polymarket": {
                    "manifest_sha256": "sha256:" + f"{index + 1:064x}",
                },
                "kalshi": {
                    "manifest_sha256": "sha256:" + f"{index + 101:064x}",
                },
            }
        )
    return {
        "index_sha256": "sha256:" + "f" * 64,
        "game_count": count,
        "games": games,
    }


def _index_path(tmp_path: Path) -> Path:
    path = tmp_path / "holdout.evidence-index.json"
    path.write_bytes(b'{"verified":"by verifier"}\n')
    return path


def _install_verifier(monkeypatch, index: dict[str, object]) -> None:
    def verify(
        index_bytes: bytes,
        *,
        store_root: str | Path,
        program_root: str | Path,
    ) -> dict[str, object]:
        assert index_bytes == b'{"verified":"by verifier"}\n'
        assert Path(store_root).name == "raw"
        assert Path(program_root).name == "program"
        return index

    monkeypatch.setattr(holdout, "verify_holdout_evidence_index", verify)


def test_selector_consumes_verified_index_and_freezes_seeded_twenty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    index = _verified_index()
    _install_verifier(monkeypatch, index)
    discovery = (str(index["games"][0]["game_id"]),)

    selected = holdout.select_historical_holdout(
        evidence_index_path=_index_path(tmp_path),
        store_root=tmp_path / "raw",
        program_root=tmp_path / "program",
        discovery_game_ids=discovery,
    )

    eligible = sorted(
        game["game_id"]
        for game in index["games"]
        if game["game_id"] not in discovery
    )
    expected = tuple(
        sorted(
            eligible,
            key=lambda game_id: (
                hashlib.sha256(f"20260725|{game_id}".encode()).hexdigest(),
                game_id,
            ),
        )[:20]
    )
    assert selected.status == "FROZEN"
    assert selected.seed == 20260725
    assert selected.selected_game_ids == expected
    assert selected.evidence_index_sha256 == index["index_sha256"]
    assert len(selected.selected_game_ids) == 20
    assert selected.selection_id.startswith("sha256:")


def test_selector_fails_closed_when_verified_index_is_invalid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def reject(*args, **kwargs):
        raise HoldoutEvidenceError("changed")

    monkeypatch.setattr(holdout, "verify_holdout_evidence_index", reject)

    selected = holdout.select_historical_holdout(
        evidence_index_path=_index_path(tmp_path),
        store_root=tmp_path / "raw",
        program_root=tmp_path / "program",
        discovery_game_ids=(),
    )

    assert selected.status == "NOT_FROZEN_MISSING_COVERAGE_EVIDENCE"
    assert selected.selected_game_ids == ()
    assert selected.evidence_index_sha256 is None
    assert "MISSING_OR_INVALID_EVIDENCE_INDEX" in selected.blockers


def test_selector_fails_closed_when_verified_eligible_pool_is_under_twenty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    index = _verified_index(count=19)
    _install_verifier(monkeypatch, index)

    selected = holdout.select_historical_holdout(
        evidence_index_path=_index_path(tmp_path),
        store_root=tmp_path / "raw",
        program_root=tmp_path / "program",
        discovery_game_ids=(),
    )

    assert selected.status == "NOT_FROZEN_MISSING_COVERAGE_EVIDENCE"
    assert selected.selected_game_ids == ()
    assert selected.eligible_game_count == 19
    assert "FEWER_THAN_20_ELIGIBLE_GAMES" in selected.blockers


def test_holdout_selection_artifact_is_content_addressed_and_deterministic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    index = _verified_index()
    _install_verifier(monkeypatch, index)
    selected = holdout.select_historical_holdout(
        evidence_index_path=_index_path(tmp_path),
        store_root=tmp_path / "raw",
        program_root=tmp_path / "program",
        discovery_game_ids=(),
    )

    first = holdout.write_historical_holdout_selection(
        selected,
        output_root=tmp_path / "first",
    )
    second = holdout.write_historical_holdout_selection(
        selected,
        output_root=tmp_path / "second",
    )

    assert first.name == second.name
    assert first.parent.name == selected.selection_id.replace(":", "-")
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text())
    assert payload["status"] == "FROZEN"
    assert len(payload["selected_game_ids"]) == 20
    assert payload["evidence_index_sha256"] == index["index_sha256"]
    assert payload["claim_boundary"]["reaction_used"] is False
