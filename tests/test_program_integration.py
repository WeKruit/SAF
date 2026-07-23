from __future__ import annotations

import csv
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

    assert "Kalshi API key" in combined
    assert "operating jurisdiction" in combined
    assert "persistent recorder" in combined
    assert "permanent data loss" in combined
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
