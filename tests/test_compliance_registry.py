from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from prediction_market.compliance import (
    ComplianceRow,
    ComplianceRegistryError,
    load_compliance_matrix,
    load_data_license_register,
    may_execute_real_money,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_registries(tmp_path: Path) -> Path:
    root = tmp_path / "program"
    (root / "registries").mkdir(parents=True)
    for name in ("compliance_matrix.csv", "data_license_register.csv"):
        shutil.copy2(PROJECT_ROOT / "registries" / name, root / "registries" / name)
    return root


def _rewrite_cell(
    path: Path,
    *,
    row_index: int,
    field: str,
    value: str,
) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    rows[row_index][field] = value
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_real_money_is_blocked_until_exact_context_is_green() -> None:
    matrix = load_compliance_matrix(PROJECT_ROOT)

    assert matrix
    assert may_execute_real_money(
        matrix,
        venue="kalshi",
        jurisdiction="UNSPECIFIED",
        account_type="UNSPECIFIED",
    ) is False
    assert all(row.status != "GREEN" for row in matrix)


def test_unknown_context_fails_closed() -> None:
    matrix = load_compliance_matrix(PROJECT_ROOT)

    assert may_execute_real_money(
        matrix,
        venue="unknown-venue",
        jurisdiction="US-IL",
        account_type="individual",
    ) is False


def test_team_i_green_is_not_sufficient_to_override_program_no_go() -> None:
    green_context = ComplianceRow(
        matrix_id="CM-999",
        venue="kalshi",
        jurisdiction="US-IL",
        account_type="individual",
        status="GREEN",
        eligibility_review="VERIFIED",
        api_terms_review="VERIFIED",
        trading_review="VERIFIED",
        evidence_as_of="2026-07-22",
        evidence_url="https://kalshi.com/docs/kalshi-member-agreement.pdf",
        open_blocker="",
        approval_ref="I-APPROVAL-TEST",
        owner="I",
        version="v0",
        due_gate="Team_I_compliance_green",
    )

    assert may_execute_real_money(
        (green_context,),
        venue="kalshi",
        jurisdiction="US-IL",
        account_type="individual",
    ) is False


def test_green_matrix_row_cannot_retain_an_open_blocker(tmp_path: Path) -> None:
    root = _copy_registries(tmp_path)
    path = root / "registries" / "compliance_matrix.csv"
    for field, value in {
        "jurisdiction": "US-IL",
        "account_type": "individual",
        "status": "GREEN",
        "eligibility_review": "VERIFIED",
        "api_terms_review": "VERIFIED",
        "trading_review": "VERIFIED",
        "approval_ref": "I-APPROVAL-TEST",
    }.items():
        _rewrite_cell(path, row_index=0, field=field, value=value)

    with pytest.raises(ComplianceRegistryError, match="GREEN.*open_blocker"):
        load_compliance_matrix(root)


def test_operational_evidence_ids_are_exactly_o001_through_o008() -> None:
    register = load_data_license_register(PROJECT_ROOT)

    assert {row.catalog_item_id for row in register} == {
        f"O-{number:03d}" for number in range(1, 9)
    }
    assert len(register) == 8
    assert all(row.status != "GREEN" for row in register)


def test_license_rows_are_evidence_dated_and_have_due_gate() -> None:
    register = load_data_license_register(PROJECT_ROOT)

    assert all(row.evidence_as_of == "2026-07-22" for row in register)
    assert all(row.evidence_url.startswith("https://") for row in register)
    assert all(row.owner == "I" for row in register)
    assert all(row.version == "v0" for row in register)
    assert all(row.due_gate == "Team_I_compliance_green" for row in register)


@pytest.mark.parametrize(
    ("filename", "field", "value", "message"),
    [
        ("compliance_matrix.csv", "status", "green", "status"),
        ("compliance_matrix.csv", "evidence_as_of", "07/22/2026", "date"),
        ("data_license_register.csv", "catalog_item_id", "O-999", "O-001"),
        ("data_license_register.csv", "evidence_url", "http://example.test", "HTTPS"),
    ],
)
def test_malformed_or_untrusted_registry_values_fail_closed(
    tmp_path: Path,
    filename: str,
    field: str,
    value: str,
    message: str,
) -> None:
    root = _copy_registries(tmp_path)
    _rewrite_cell(root / "registries" / filename, row_index=0, field=field, value=value)

    with pytest.raises(ComplianceRegistryError, match=message):
        if filename == "compliance_matrix.csv":
            load_compliance_matrix(root)
        else:
            load_data_license_register(root)


def test_duplicate_matrix_context_fails_closed(tmp_path: Path) -> None:
    root = _copy_registries(tmp_path)
    matrix_path = root / "registries" / "compliance_matrix.csv"
    lines = matrix_path.read_text(encoding="utf-8").splitlines()
    duplicate_context = lines[1].replace("CM-001", "CM-099", 1)
    matrix_path.write_text(
        "\n".join([*lines, duplicate_context]) + "\n", encoding="utf-8"
    )

    with pytest.raises(ComplianceRegistryError, match="duplicate context"):
        load_compliance_matrix(root)


def test_extra_columns_fail_closed(tmp_path: Path) -> None:
    root = _copy_registries(tmp_path)
    register_path = root / "registries" / "data_license_register.csv"
    lines = register_path.read_text(encoding="utf-8").splitlines()
    register_path.write_text(
        "\n".join([lines[0] + ",hidden", *[line + ",value" for line in lines[1:]]])
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ComplianceRegistryError, match="columns"):
        load_data_license_register(root)


def test_license_green_requires_consistent_permissions_and_approval(
    tmp_path: Path,
) -> None:
    root = _copy_registries(tmp_path)
    path = root / "registries" / "data_license_register.csv"
    for field, value in {
        "status": "GREEN",
        "commercial_use": "PROHIBITED",
        "redistribution": "PERMITTED",
        "attribution_required": "YES",
        "operational_use": "APPROVED",
        "open_blocker": "",
        "approval_ref": "I-APPROVAL-TEST",
    }.items():
        _rewrite_cell(path, row_index=0, field=field, value=value)

    with pytest.raises(ComplianceRegistryError, match="GREEN.*commercial_use"):
        load_data_license_register(root)
