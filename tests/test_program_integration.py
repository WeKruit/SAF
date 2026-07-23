from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from prediction_market.program_audit import (
    audit_no_go,
    load_artifact_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rows(relative: str) -> list[dict[str, str]]:
    with (PROJECT_ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_all_first_round_artifacts_have_owner_version_due_gate_and_real_path() -> None:
    rows = load_artifact_registry(PROJECT_ROOT)

    assert len(rows) >= 45
    assert all(row.owner_team and row.version and row.due_gate for row in rows)
    assert all((PROJECT_ROOT / row.path).is_file() for row in rows)
    assert {row.status for row in rows} <= {
        "registered",
        "complete",
        "blocked",
        "in_progress",
        "harness_pass",
        "PRELIMINARY_RESEARCH_ONLY",
        "POC_ONLY",
    }


def test_week_one_backlog_covers_every_team_and_has_explicit_status() -> None:
    rows = _rows("artifacts/architecture/week1_backlog_v0.csv")

    teams = {team[0] for row in rows for team in row["team"].split("+")}
    assert set("ABCDEFGHI") <= teams
    assert all(row["owner"] and row["version"] and row["due_gate"] for row in rows)
    assert all(row["status"] in {"COMPLETE", "IN_PROGRESS", "BLOCKED"} for row in rows)


def test_risk_register_exposes_required_credentials_and_permanent_data_loss() -> None:
    rows = _rows("artifacts/architecture/risk_blocker_register_v0.csv")
    combined = "\n".join(" ".join(row.values()) for row in rows)
    by_id = {row["risk_id"]: row for row in rows}

    assert "Kalshi API key" in combined
    assert "operating jurisdiction" in combined
    assert "persistent recorder" in combined
    assert "permanent data loss" in combined
    assert "X-02 formal registry acceptance" in combined
    assert "X-01 Phase 1 reconstruction pipeline" in combined
    assert by_id["RB-004"]["blocked_scope"] == (
        "X-01 independent comparison and X-03 four-week sport storage"
    )
    assert "independent comparison source" in by_id["RB-004"]["required_input"]
    assert "X-03" in by_id["RB-004"]["required_input"]
    assert "Full PMXT Phase 0" not in by_id["RB-004"]["blocked_scope"]
    assert all(row["owner"] and row["version"] and row["due_gate"] for row in rows)


def test_no_go_audit_remains_closed() -> None:
    report = audit_no_go(PROJECT_ROOT)

    assert report.violations == ()
    assert report.real_money_authorized is False
    assert report.live_maker_present is False
    assert report.live_arbitrage_present is False
    assert report.llm_hot_path_present is False


def test_program_status_is_conditional_and_never_claims_blocked_experiments_done() -> None:
    text = (
        PROJECT_ROOT / "artifacts" / "architecture" / "program_status_v0.md"
    ).read_text(encoding="utf-8")

    assert "CONDITIONAL_GO" in text
    assert "NO_REAL_MONEY" in text
    assert "X-09" in text and "EXPERIMENT_BLOCKED" in text
    assert "X-10" in text and "REGISTERED_NOT_RUN" in text
    assert "PMXT L2" in text and "queue fill" in text
    assert "X-01: `DATA_READY_PREFLIGHT_ONLY`" in text
    assert "24-hour" in text
    assert "`reconstruction_executed: false`" in text
    assert "X-02: `DATA_READY_NO_RESULT`" in text
    assert "four selected UTC days" in text
    assert "`results_ref` remains empty" in text
    assert "X-11: `PRELIMINARY_POC`" in text
    assert "`PIT_UNPROVEN`" in text
    assert "X-12: `POC_ONLY`" in text
    assert "15-second" in text and "five-frame" in text
    assert "seven-day gate is not met" in text


def test_week_one_backlog_tracks_current_data_and_poc_boundaries() -> None:
    rows = _rows("artifacts/architecture/week1_backlog_v0.csv")
    by_id = {row["backlog_id"]: row for row in rows}

    assert "24-hour" in by_id["W1-007"]["blocker"]
    assert "all-contract reconstruction" in by_id["W1-007"]["blocker"]
    assert "independent comparison" in by_id["W1-007"]["blocker"]
    assert "lacks a full game day" not in by_id["W1-007"]["blocker"]
    assert by_id["W1-012"]["deliverable"] == (
        "artifacts/game-state/nfl/x11_real_data_pipeline_evidence_v0.json"
    )
    assert "PIT_UNPROVEN" in by_id["W1-012"]["blocker"]
    assert by_id["W1-013"]["deliverable"] == (
        "artifacts/game-state/soccer/x12_real_data_poc_v0.json"
    )
    assert "POC_ONLY" in by_id["W1-013"]["blocker"]
    assert "experiment ID" not in by_id["W1-012"]["blocker"]
    assert "experiment ID" not in by_id["W1-013"]["blocker"]


def test_current_governance_evidence_is_registered_without_promotion() -> None:
    expected = {
        "artifacts/data-audit/x01_full_day_input_manifest_v1.json": {
            "artifact_id": "ART-C-016",
            "owner_team": "C+H",
            "version": "v1",
            "due_gate": "X01_L2_reconstruction",
            "status": "complete",
        },
        "artifacts/data-audit/x01_full_day_preflight_v1.json": {
            "artifact_id": "ART-C-017",
            "owner_team": "C+H",
            "version": "v1",
            "due_gate": "X01_L2_reconstruction",
            "status": "complete",
        },
        "artifacts/data-audit/kalshi_public_backfill_v1.json": {
            "artifact_id": "ART-C-018",
            "owner_team": "C",
            "version": "v1",
            "due_gate": "2026-07-29_W1_data_ready",
            "status": "in_progress",
        },
        "artifacts/venue-connectivity/polymarket_recorder_supervisor_v1.md": {
            "artifact_id": "ART-B-006",
            "owner_team": "B+C",
            "version": "v1",
            "due_gate": "X08_prospective_recorder_continuity",
            "status": "complete",
        },
        "artifacts/venue-connectivity/polymarket_recorder_smoke_20260722_v1.json": {
            "artifact_id": "ART-B-007",
            "owner_team": "B+C",
            "version": "v1",
            "due_gate": "X08_prospective_recorder_continuity",
            "status": "PRELIMINARY_RESEARCH_ONLY",
        },
        "artifacts/game-state/nfl/x11_real_data_pipeline_evidence_v0.json": {
            "artifact_id": "ART-D2-002",
            "owner_team": "D2+H",
            "version": "v0",
            "due_gate": "2026-08-05_W2_review",
            "status": "PRELIMINARY_RESEARCH_ONLY",
        },
        "artifacts/game-state/mlb/source_inventory_v0.json": {
            "artifact_id": "ART-D4-001",
            "owner_team": "D4+I",
            "version": "v0",
            "due_gate": "2026-08-05_W2_review",
            "status": "PRELIMINARY_RESEARCH_ONLY",
        },
        "artifacts/game-state/f1/source_inventory_v0.json": {
            "artifact_id": "ART-D5-001",
            "owner_team": "D5+I",
            "version": "v0",
            "due_gate": "2026-08-05_W2_review",
            "status": "PRELIMINARY_RESEARCH_ONLY",
        },
    }
    rows = {
        row["path"]: {
            key: value for key, value in row.items() if key != "path"
        }
        for row in _rows("registries/artifact_registry.csv")
    }

    assert {path: rows[path] for path in expected} == expected

    kalshi = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/data-audit/kalshi_public_backfill_v1.json"
        ).read_text(encoding="utf-8")
    )
    claimed_kalshi_hash = kalshi.pop("artifact_sha256")
    assert claimed_kalshi_hash == (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                kalshi,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )
    assert kalshi["status"] == "BOUNDED_INCOMPLETE_LICENSE_PENDING"
    assert kalshi["verification"]["main_page_manifests_verified"] == 200
    assert kalshi["verification"]["main_page_objects_verified"] == 200
    assert {
        resource: (
            run["page_count"],
            run["unique_item_count"],
            run["complete"],
            run["terminal_cursor_empty"],
        )
        for resource, run in kalshi["runs"].items()
    } == {
        "markets": (100, 100_000, False, False),
        "trades": (100, 100_000, False, False),
    }
    assert kalshi["boundaries"] == {
        "complete": False,
        "historical_l2_claimed": False,
        "license_review_id": "O-003",
        "license_status": "pending",
        "formal_result_eligible": False,
        "overlap_claimed": False,
        "live_historical_overlap_result": "NOT_RUN",
    }

    observation = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/venue-connectivity/polymarket_recorder_smoke_20260722_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert observation["formal_x08_result"] is False
    assert observation["health"]["frames"] == 5
    assert observation["prospective_observation"] == {
        "duration_gate_met": False,
        "fixtures_can_satisfy_elapsed_time": False,
        "observed_elapsed_days": 0.00017361111111111112,
        "observed_elapsed_seconds": 15.0,
        "required_elapsed_days": 7,
    }

    x11 = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/game-state/nfl/x11_real_data_pipeline_evidence_v0.json"
        ).read_text(encoding="utf-8")
    )
    assert x11["execution_mode"] == "full"
    assert x11["result_label"] == "PRELIMINARY"
    assert x11["pit_assessment"]["method_status"] == "PIT_UNPROVEN"
    assert x11["formal_result_eligible"] is False


def test_x12_real_data_artifact_is_registered_as_poc_only() -> None:
    relative = "artifacts/game-state/soccer/x12_real_data_poc_v0.json"
    artifact = next(
        row
        for row in _rows("registries/artifact_registry.csv")
        if row["path"] == relative
    )
    assert artifact == {
        "artifact_id": "ART-D3-002",
        "path": relative,
        "owner_team": "D3+H+I",
        "version": "v0",
        "due_gate": "2026-08-05_W2_review",
        "status": "POC_ONLY",
    }

    evidence = json.loads((PROJECT_ROOT / relative).read_bytes())
    claimed_self_hash = evidence.pop("evidence_sha256")
    canonical = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert claimed_self_hash == (
        "sha256:" + hashlib.sha256(canonical).hexdigest()
    )
    assert evidence["promotion_decision"] == "POC_ONLY"
    assert evidence["is_formal_result"] is False
    assert evidence["formal_result_eligible"] is False
    assert evidence["market_prior"] == {
        "available": False,
        "imputed": False,
        "reason": "no_point_in_time_market_prior",
    }
