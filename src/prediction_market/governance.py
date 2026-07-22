"""Validation for the research program's governed source files."""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REQUIRED_SOURCE_FILENAMES = (
    "research_program_charter_v0.2.md",
    "catalog_registry.csv",
    "catalog_team_assignments.csv",
)
EXPECTED_CATALOG_COUNT = 87
EXPECTED_ASSIGNMENT_COUNT = 150
ALLOWED_RESPONSIBILITIES = frozenset({"primary", "secondary"})


@dataclass(frozen=True, slots=True)
class GovernanceViolation:
    """One machine-readable governance rule violation."""

    code: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class GovernanceReport:
    """Immutable result of validating the governed program sources."""

    source_files: tuple[str, ...]
    source_count: int
    catalog_count: int
    assignment_count: int
    violations: tuple[GovernanceViolation, ...]

    @property
    def is_valid(self) -> bool:
        return not self.violations


def _add_violation(
    violations: list[GovernanceViolation],
    code: str,
    path: str,
    message: str,
) -> None:
    violations.append(GovernanceViolation(code=code, path=path, message=message))


def _read_utf8_text(
    path: Path,
    relative_path: str,
    violation_prefix: str,
    violations: list[GovernanceViolation],
) -> str | None:
    try:
        contents = path.read_bytes()
    except OSError as error:
        _add_violation(
            violations,
            f"{violation_prefix}.read_error",
            relative_path,
            f"could not read file: {error}",
        )
        return None
    try:
        return contents.decode("utf-8")
    except UnicodeDecodeError as error:
        _add_violation(
            violations,
            f"{violation_prefix}.decode_error",
            relative_path,
            f"file is not valid UTF-8: {error}",
        )
        return None


def _read_manifest(
    manifest_path: Path,
    violations: list[GovernanceViolation],
) -> tuple[tuple[str, str], ...]:
    relative_path = "charter/SOURCE_MANIFEST.sha256"
    if not manifest_path.is_file():
        _add_violation(
            violations,
            "manifest.missing",
            relative_path,
            "source manifest is missing",
        )
        return ()

    manifest_text = _read_utf8_text(
        manifest_path, relative_path, "manifest", violations
    )
    if manifest_text is None:
        return ()

    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        manifest_text.splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            _add_violation(
                violations,
                "manifest.invalid_entry",
                relative_path,
                f"line {line_number} must contain a SHA-256 digest and filename",
            )
            continue
        digest, filename = parts
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            _add_violation(
                violations,
                "manifest.invalid_digest",
                relative_path,
                f"line {line_number} does not contain a lowercase SHA-256 digest",
            )
            continue
        entries.append((digest, filename))
    return tuple(entries)


def _read_csv_rows(
    path: Path,
    relative_path: str,
    required_columns: frozenset[str],
    violation_prefix: str,
    violations: list[GovernanceViolation],
) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        _add_violation(
            violations,
            f"{violation_prefix}.missing",
            relative_path,
            "required CSV file is missing",
        )
        return ()

    csv_text = _read_utf8_text(
        path, relative_path, violation_prefix, violations
    )
    if csv_text is None:
        return ()

    with io.StringIO(csv_text, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = frozenset(reader.fieldnames or ())
        missing_columns = sorted(required_columns - fieldnames)
        if missing_columns:
            _add_violation(
                violations,
                f"{violation_prefix}.missing_columns",
                relative_path,
                f"missing columns: {', '.join(missing_columns)}",
            )
        rows: list[dict[str, str]] = []
        for row in reader:
            missing_values = sorted(
                column
                for column in required_columns & fieldnames
                if row.get(column) is None or not row[column].strip()
            )
            if missing_values:
                _add_violation(
                    violations,
                    f"{violation_prefix}.malformed_row",
                    relative_path,
                    f"line {reader.line_num} is missing values for: "
                    f"{', '.join(missing_values)}",
                )
            rows.append(
                {
                    key: value if value is not None else ""
                    for key, value in row.items()
                    if key is not None
                }
            )
        return tuple(rows)


def validate_program(root: str | Path) -> GovernanceReport:
    """Validate the program source manifest, catalog, and team assignments."""

    program_root = Path(root)
    charter_dir = program_root / "charter"
    violations: list[GovernanceViolation] = []

    manifest_entries = _read_manifest(
        charter_dir / "SOURCE_MANIFEST.sha256", violations
    )
    source_files = tuple(filename for _, filename in manifest_entries)
    if Counter(source_files) != Counter(REQUIRED_SOURCE_FILENAMES):
        _add_violation(
            violations,
            "manifest.source_files",
            "charter/SOURCE_MANIFEST.sha256",
            "manifest must list exactly the three program source filenames",
        )

    manifest_digests = {filename: digest for digest, filename in manifest_entries}
    for filename in REQUIRED_SOURCE_FILENAMES:
        source_path = charter_dir / filename
        relative_path = f"charter/{filename}"
        if not source_path.is_file():
            _add_violation(
                violations,
                "source.missing",
                relative_path,
                "required program source is missing",
            )
            continue
        try:
            source_bytes = source_path.read_bytes()
        except OSError as error:
            _add_violation(
                violations,
                "source.read_error",
                relative_path,
                f"could not read file: {error}",
            )
            continue
        actual_digest = hashlib.sha256(source_bytes).hexdigest()
        expected_digest = manifest_digests.get(filename)
        if expected_digest is not None and actual_digest != expected_digest:
            _add_violation(
                violations,
                "source.hash_mismatch",
                relative_path,
                f"expected {expected_digest}, found {actual_digest}",
            )
        try:
            source_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            _add_violation(
                violations,
                "source.decode_error",
                relative_path,
                f"file is not valid UTF-8: {error}",
            )

    catalog_rows = _read_csv_rows(
        charter_dir / "catalog_registry.csv",
        "charter/catalog_registry.csv",
        frozenset({"catalog_item_id"}),
        "catalog",
        violations,
    )
    catalog_ids = [row.get("catalog_item_id", "").strip() for row in catalog_rows]
    if len(catalog_rows) != EXPECTED_CATALOG_COUNT:
        _add_violation(
            violations,
            "catalog.row_count",
            "charter/catalog_registry.csv",
            f"expected {EXPECTED_CATALOG_COUNT} rows, found {len(catalog_rows)}",
        )
    duplicate_catalog_ids = sorted(
        catalog_id
        for catalog_id, count in Counter(catalog_ids).items()
        if count > 1
    )
    if duplicate_catalog_ids:
        _add_violation(
            violations,
            "catalog.duplicate_id",
            "charter/catalog_registry.csv",
            f"duplicate catalog IDs: {', '.join(duplicate_catalog_ids)}",
        )

    assignment_rows = _read_csv_rows(
        charter_dir / "catalog_team_assignments.csv",
        "charter/catalog_team_assignments.csv",
        frozenset({"catalog_item_id", "team", "responsibility"}),
        "assignments",
        violations,
    )
    if len(assignment_rows) != EXPECTED_ASSIGNMENT_COUNT:
        _add_violation(
            violations,
            "assignments.row_count",
            "charter/catalog_team_assignments.csv",
            f"expected {EXPECTED_ASSIGNMENT_COUNT} rows, found {len(assignment_rows)}",
        )

    invalid_responsibilities = sorted(
        {
            responsibility
            for row in assignment_rows
            if (responsibility := row.get("responsibility", "")).strip()
            and responsibility not in ALLOWED_RESPONSIBILITIES
        }
    )
    if invalid_responsibilities:
        _add_violation(
            violations,
            "assignments.invalid_responsibility",
            "charter/catalog_team_assignments.csv",
            "responsibility must be exactly primary or secondary; found: "
            + ", ".join(invalid_responsibilities),
        )

    known_catalog_ids = set(catalog_ids)
    assigned_ids = {
        row.get("catalog_item_id", "").strip() for row in assignment_rows
    }
    unknown_catalog_ids = sorted(assigned_ids - known_catalog_ids)
    if unknown_catalog_ids:
        _add_violation(
            violations,
            "assignments.unknown_catalog_id",
            "charter/catalog_team_assignments.csv",
            f"unknown catalog IDs: {', '.join(unknown_catalog_ids)}",
        )

    primary_counts = Counter(
        row.get("catalog_item_id", "").strip()
        for row in assignment_rows
        if row.get("responsibility", "") == "primary"
    )
    invalid_primary_counts = sorted(
        (catalog_id, primary_counts[catalog_id])
        for catalog_id in known_catalog_ids
        if primary_counts[catalog_id] != 1
    )
    if invalid_primary_counts:
        details = ", ".join(
            f"{catalog_id}={count}" for catalog_id, count in invalid_primary_counts
        )
        _add_violation(
            violations,
            "assignments.primary_count",
            "charter/catalog_team_assignments.csv",
            f"catalog items must have exactly one primary assignment: {details}",
        )

    return GovernanceReport(
        source_files=source_files,
        source_count=len(manifest_entries),
        catalog_count=len(catalog_rows),
        assignment_count=len(assignment_rows),
        violations=tuple(violations),
    )
