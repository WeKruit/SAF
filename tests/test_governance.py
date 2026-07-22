from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHARTER_DIR = PROJECT_ROOT / "charter"
SOURCE_MANIFEST = CHARTER_DIR / "SOURCE_MANIFEST.sha256"
REQUIRED_SOURCE_FILENAMES = (
    "research_program_charter_v0.2.md",
    "catalog_registry.csv",
    "catalog_team_assignments.csv",
)
VALIDATOR_SCRIPT = PROJECT_ROOT / "tools" / "validate_governance.py"

sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _manifest_entries(manifest: Path = SOURCE_MANIFEST) -> list[tuple[str, str]]:
    return [
        tuple(line.split(maxsplit=1))
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate(root: Path = PROJECT_ROOT):
    try:
        from prediction_market.governance import validate_program
    except ModuleNotFoundError:
        pytest.fail("prediction_market.governance has not been implemented")
    return validate_program(root)


def _corrupted_fixture(tmp_path: Path) -> Path:
    fixture_root = tmp_path / "program"
    shutil.copytree(CHARTER_DIR, fixture_root / "charter")
    registry = fixture_root / "charter" / "catalog_registry.csv"
    registry.write_text(
        registry.read_text(encoding="utf-8").replace("R-001", "R-999", 1),
        encoding="utf-8",
    )
    return fixture_root


def _malformed_assignment_fixture(tmp_path: Path) -> Path:
    fixture_root = tmp_path / "malformed-program"
    shutil.copytree(CHARTER_DIR, fixture_root / "charter")
    assignments = fixture_root / "charter" / "catalog_team_assignments.csv"
    lines = assignments.read_text(encoding="utf-8").splitlines()
    lines[-1] = "O-008"
    assignments.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fixture_root


def _fixture_with_csv_cell(
    tmp_path: Path,
    filename: str,
    row_index: int,
    field: str,
    value: str,
) -> Path:
    fixture_root = tmp_path / "cell-program"
    shutil.copytree(CHARTER_DIR, fixture_root / "charter")
    csv_path = fixture_root / "charter" / filename
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    rows[row_index][field] = value
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return fixture_root


def _fixture_with_invalid_utf8(tmp_path: Path, relative_path: str) -> Path:
    fixture_root = tmp_path / "invalid-utf8-program"
    shutil.copytree(CHARTER_DIR, fixture_root / "charter")
    target = fixture_root / relative_path
    target.write_bytes(target.read_bytes() + b"\xff")
    return fixture_root


def _run_validator(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR_SCRIPT), str(root)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_source_manifest_hashes_all_program_sources() -> None:
    for expected_digest, filename in _manifest_entries():
        actual_digest = hashlib.sha256((CHARTER_DIR / filename).read_bytes()).hexdigest()
        assert actual_digest == expected_digest

    report = _validate()
    assert report.is_valid
    assert report.source_count == 3


def test_source_manifest_contains_exact_required_filenames() -> None:
    filenames = tuple(filename for _, filename in _manifest_entries())
    assert filenames == REQUIRED_SOURCE_FILENAMES

    report = _validate()
    assert report.source_files == REQUIRED_SOURCE_FILENAMES


def test_catalog_contains_exactly_87_unique_ids() -> None:
    catalog_rows = _csv_rows(CHARTER_DIR / "catalog_registry.csv")
    catalog_ids = [row["catalog_item_id"] for row in catalog_rows]
    assert len(catalog_rows) == 87
    assert len(set(catalog_ids)) == 87

    report = _validate()
    assert report.catalog_count == 87


def test_assignment_table_contains_exactly_150_rows() -> None:
    assignment_rows = _csv_rows(CHARTER_DIR / "catalog_team_assignments.csv")
    assert len(assignment_rows) == 150

    report = _validate()
    assert report.assignment_count == 150


def test_all_assignments_reference_known_catalog_ids() -> None:
    catalog_ids = {
        row["catalog_item_id"]
        for row in _csv_rows(CHARTER_DIR / "catalog_registry.csv")
    }
    assigned_ids = {
        row["catalog_item_id"]
        for row in _csv_rows(CHARTER_DIR / "catalog_team_assignments.csv")
    }
    assert assigned_ids <= catalog_ids

    assert _validate().is_valid


def test_every_catalog_item_has_exactly_one_primary_assignment() -> None:
    catalog_ids = {
        row["catalog_item_id"]
        for row in _csv_rows(CHARTER_DIR / "catalog_registry.csv")
    }
    primary_counts = Counter(
        row["catalog_item_id"]
        for row in _csv_rows(CHARTER_DIR / "catalog_team_assignments.csv")
        if row["responsibility"] == "primary"
    )
    assert {catalog_id: primary_counts[catalog_id] for catalog_id in catalog_ids} == {
        catalog_id: 1 for catalog_id in catalog_ids
    }

    assert _validate().is_valid


def test_validator_returns_immutable_structured_violations(
    tmp_path: Path,
) -> None:
    from prediction_market.governance import GovernanceViolation

    report = _validate(_corrupted_fixture(tmp_path))

    assert not report.is_valid
    assert isinstance(report.violations, tuple)
    assert all(isinstance(violation, GovernanceViolation) for violation in report.violations)
    assert {
        "source.hash_mismatch",
        "assignments.unknown_catalog_id",
        "assignments.primary_count",
    } <= {violation.code for violation in report.violations}
    with pytest.raises(FrozenInstanceError):
        report.catalog_count = 0
    with pytest.raises(FrozenInstanceError):
        report.violations[0].code = "changed"


def test_cli_exits_nonzero_for_corrupted_fixture(tmp_path: Path) -> None:
    result = _run_validator(_corrupted_fixture(tmp_path))

    assert result.returncode == 1
    assert "source.hash_mismatch" in result.stdout


def test_malformed_assignment_returns_structured_violation_without_traceback(
    tmp_path: Path,
) -> None:
    fixture_root = _malformed_assignment_fixture(tmp_path)

    report = _validate(fixture_root)
    result = _run_validator(fixture_root)

    assert "assignments.malformed_row" in {
        violation.code for violation in report.violations
    }
    assert result.returncode == 1
    assert "assignments.malformed_row" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("filename", "row_index", "field", "value", "violation_code"),
    [
        ("catalog_registry.csv", 0, "catalog_item_id", "", "catalog.malformed_row"),
        (
            "catalog_registry.csv",
            0,
            "catalog_item_id",
            "  ",
            "catalog.malformed_row",
        ),
        (
            "catalog_team_assignments.csv",
            1,
            "catalog_item_id",
            "",
            "assignments.malformed_row",
        ),
        (
            "catalog_team_assignments.csv",
            1,
            "catalog_item_id",
            "  ",
            "assignments.malformed_row",
        ),
        (
            "catalog_team_assignments.csv",
            1,
            "team",
            "",
            "assignments.malformed_row",
        ),
        (
            "catalog_team_assignments.csv",
            1,
            "team",
            "  ",
            "assignments.malformed_row",
        ),
        (
            "catalog_team_assignments.csv",
            1,
            "responsibility",
            "",
            "assignments.malformed_row",
        ),
        (
            "catalog_team_assignments.csv",
            1,
            "responsibility",
            "  ",
            "assignments.malformed_row",
        ),
    ],
    ids=(
        "empty-catalog-id",
        "whitespace-catalog-id",
        "empty-assignment-id",
        "whitespace-assignment-id",
        "empty-team",
        "whitespace-team",
        "empty-responsibility",
        "whitespace-responsibility",
    ),
)
def test_blank_required_csv_cells_are_structured_violations(
    tmp_path: Path,
    filename: str,
    row_index: int,
    field: str,
    value: str,
    violation_code: str,
) -> None:
    fixture_root = _fixture_with_csv_cell(
        tmp_path, filename, row_index, field, value
    )

    report = _validate(fixture_root)
    result = _run_validator(fixture_root)

    assert violation_code in {violation.code for violation in report.violations}
    assert result.returncode == 1
    assert violation_code in result.stdout
    assert result.stderr == ""


def test_assignment_responsibility_rejects_unknown_token(tmp_path: Path) -> None:
    fixture_root = _fixture_with_csv_cell(
        tmp_path,
        "catalog_team_assignments.csv",
        1,
        "responsibility",
        "owner",
    )

    report = _validate(fixture_root)
    result = _run_validator(fixture_root)

    assert "assignments.invalid_responsibility" in {
        violation.code for violation in report.violations
    }
    assert result.returncode == 1
    assert "assignments.invalid_responsibility" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("relative_path", "violation_code", "expects_hash_mismatch"),
    [
        ("charter/SOURCE_MANIFEST.sha256", "manifest.decode_error", False),
        ("charter/research_program_charter_v0.2.md", "source.decode_error", True),
        ("charter/catalog_registry.csv", "catalog.decode_error", True),
        (
            "charter/catalog_team_assignments.csv",
            "assignments.decode_error",
            True,
        ),
    ],
    ids=("manifest", "charter", "catalog", "assignments"),
)
def test_invalid_utf8_returns_structured_violations_without_traceback(
    tmp_path: Path,
    relative_path: str,
    violation_code: str,
    expects_hash_mismatch: bool,
) -> None:
    fixture_root = _fixture_with_invalid_utf8(tmp_path, relative_path)

    report = _validate(fixture_root)
    result = _run_validator(fixture_root)
    violation_codes = {violation.code for violation in report.violations}

    assert violation_code in violation_codes
    if expects_hash_mismatch:
        assert "source.hash_mismatch" in violation_codes
        assert "source.hash_mismatch" in result.stdout
    assert result.returncode == 1
    assert violation_code in result.stdout
    assert result.stderr == ""


def test_oserror_returns_structured_read_violations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open

    def failing_open(path: Path, *args, **kwargs):
        if path.name == "catalog_team_assignments.csv":
            raise OSError("injected read failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    report = _validate()
    violation_codes = {violation.code for violation in report.violations}

    assert "source.read_error" in violation_codes
    assert "assignments.read_error" in violation_codes


def test_cli_prints_exact_concise_success() -> None:
    result = _run_validator(PROJECT_ROOT)

    assert result.returncode == 0
    assert result.stdout == "OK: 3 sources, 87 catalog items, 150 assignments\n"
    assert result.stderr == ""
