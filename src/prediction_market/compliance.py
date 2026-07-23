"""Fail-closed loaders for Team I compliance and licensing gates."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit


COMPLIANCE_COLUMNS = (
    "matrix_id",
    "venue",
    "jurisdiction",
    "account_type",
    "status",
    "eligibility_review",
    "api_terms_review",
    "trading_review",
    "evidence_as_of",
    "evidence_url",
    "open_blocker",
    "approval_ref",
    "owner",
    "version",
    "due_gate",
)
LICENSE_COLUMNS = (
    "catalog_item_id",
    "source_name",
    "status",
    "commercial_use",
    "redistribution",
    "attribution_required",
    "operational_use",
    "evidence_as_of",
    "evidence_url",
    "open_blocker",
    "approval_ref",
    "owner",
    "version",
    "due_gate",
)
COMPLIANCE_STATUSES = frozenset({"NOT_GREEN_OPEN", "NOT_GREEN_BLOCKED", "GREEN"})
REVIEW_STATUSES = frozenset({"OPEN", "BLOCKED", "VERIFIED"})
USE_STATUSES = frozenset(
    {"UNKNOWN", "PROHIBITED", "PERMITTED", "PERMITTED_WITH_CONDITIONS"}
)
ATTRIBUTION_STATUSES = frozenset({"UNKNOWN", "YES", "NO"})
OPERATIONAL_USE_STATUSES = frozenset({"UNKNOWN", "RESEARCH_ONLY", "BLOCKED", "APPROVED"})
EXPECTED_LICENSE_REVIEW_IDS = (
    *(f"O-{number:03d}" for number in range(1, 9)),
    "R-039",
    "I-018",
    "R-042",
    "R-043",
)
CATALOG_COLUMNS = (
    "catalog_item_id",
    "source_catalog_id",
    "catalog",
    "title",
    "primary_team",
    "secondary_teams",
    "priority",
    "program_stage",
    "first_artifact",
    "linked_experiments",
    "status",
    "due_gate",
)
_MATRIX_ID = re.compile(r"CM-[0-9]{3}")
_CATALOG_ID = re.compile(r"(?:R|I|O)-[0-9]{3}")
_APPROVAL_REF = re.compile(r"I-APPROVAL-[A-Z0-9][A-Z0-9._-]*")
_VERSION = re.compile(r"v[0-9]+(?:\.[0-9]+)?")


class ComplianceRegistryError(ValueError):
    """A compliance registry is missing, malformed, or unsafe to trust."""


@dataclass(frozen=True, slots=True)
class ComplianceRow:
    matrix_id: str
    venue: str
    jurisdiction: str
    account_type: str
    status: str
    eligibility_review: str
    api_terms_review: str
    trading_review: str
    evidence_as_of: str
    evidence_url: str
    open_blocker: str
    approval_ref: str
    owner: str
    version: str
    due_gate: str


@dataclass(frozen=True, slots=True)
class DataLicenseRow:
    catalog_item_id: str
    source_name: str
    status: str
    commercial_use: str
    redistribution: str
    attribution_required: str
    operational_use: str
    evidence_as_of: str
    evidence_url: str
    open_blocker: str
    approval_ref: str
    owner: str
    version: str
    due_gate: str


def _safe_registry_path(root: str | Path, filename: str) -> Path:
    program_root = Path(root).resolve()
    registries = (program_root / "registries").resolve()
    path = program_root / "registries" / filename
    if path.is_symlink():
        raise ComplianceRegistryError(f"{filename} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(registries)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ComplianceRegistryError(f"unsafe or missing registry: {filename}") from error
    if not resolved.is_file():
        raise ComplianceRegistryError(f"registry is not a regular file: {filename}")
    return resolved


def _stable_catalog_ids(root: str | Path) -> set[str]:
    program_root = Path(root).resolve()
    charter_root = (program_root / "charter").resolve()
    path = program_root / "charter" / "catalog_registry.csv"
    if path.is_symlink():
        raise ComplianceRegistryError(
            "catalog_registry.csv must not be a symlink"
        )
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(charter_root)
        text = resolved.read_bytes().decode("utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError) as error:
        raise ComplianceRegistryError(
            "unsafe or unreadable stable catalog registry"
        ) from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if tuple(reader.fieldnames or ()) != CATALOG_COLUMNS:
        raise ComplianceRegistryError(
            "stable catalog registry columns do not match contract"
        )
    identifiers: set[str] = set()
    try:
        for row in reader:
            identifier = row.get("catalog_item_id")
            if (
                None in row
                or type(identifier) is not str
                or not _CATALOG_ID.fullmatch(identifier)
                or identifier in identifiers
            ):
                raise ComplianceRegistryError(
                    "stable catalog registry has invalid or duplicate IDs"
                )
            identifiers.add(identifier)
    except csv.Error as error:
        raise ComplianceRegistryError(
            "stable catalog registry is malformed"
        ) from error
    return identifiers


def _read_csv(
    root: str | Path,
    filename: str,
    expected_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    path = _safe_registry_path(root, filename)
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ComplianceRegistryError(f"cannot read UTF-8 registry {filename}") from error
    if "\x00" in text:
        raise ComplianceRegistryError(f"NUL byte in registry {filename}")
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if tuple(reader.fieldnames or ()) != expected_columns:
        raise ComplianceRegistryError(
            f"{filename} columns must be exactly: {', '.join(expected_columns)}"
        )
    rows: list[dict[str, str]] = []
    try:
        for row in reader:
            if None in row:
                raise ComplianceRegistryError(f"unexpected columns on line {reader.line_num}")
            normalized: dict[str, str] = {}
            for field in expected_columns:
                value = row.get(field)
                if value is None or value != value.strip():
                    raise ComplianceRegistryError(
                        f"non-canonical value for {field} on line {reader.line_num}"
                    )
                normalized[field] = value
            rows.append(normalized)
    except csv.Error as error:
        raise ComplianceRegistryError(f"invalid CSV in {filename}: {error}") from error
    if not rows:
        raise ComplianceRegistryError(f"{filename} must not be empty")
    return rows


def _validate_date(value: str) -> None:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ComplianceRegistryError(f"invalid evidence date: {value}") from error
    if parsed.isoformat() != value:
        raise ComplianceRegistryError(f"non-canonical evidence date: {value}")


def _validate_https_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ComplianceRegistryError(f"evidence URL must be canonical HTTPS: {value}")


def _validate_common(row: dict[str, str]) -> None:
    if row["status"] not in COMPLIANCE_STATUSES:
        raise ComplianceRegistryError(f"invalid status: {row['status']}")
    _validate_date(row["evidence_as_of"])
    _validate_https_url(row["evidence_url"])
    if row["owner"] != "I":
        raise ComplianceRegistryError("Team I must own every compliance row")
    if not _VERSION.fullmatch(row["version"]):
        raise ComplianceRegistryError(f"invalid version: {row['version']}")
    if row["due_gate"] != "Team_I_compliance_green":
        raise ComplianceRegistryError("invalid due gate")
    if row["status"] == "GREEN":
        if row["open_blocker"]:
            raise ComplianceRegistryError("GREEN row cannot retain open_blocker")
    elif not row["open_blocker"]:
        raise ComplianceRegistryError("non-GREEN open_blocker must be explicit")


def load_compliance_matrix(root: str | Path) -> tuple[ComplianceRow, ...]:
    """Load the venue/context matrix and reject ambiguous or unsafe rows."""

    raw_rows = _read_csv(root, "compliance_matrix.csv", COMPLIANCE_COLUMNS)
    rows: list[ComplianceRow] = []
    matrix_ids: set[str] = set()
    contexts: set[tuple[str, str, str]] = set()
    for raw in raw_rows:
        _validate_common(raw)
        if not _MATRIX_ID.fullmatch(raw["matrix_id"]):
            raise ComplianceRegistryError(f"invalid matrix_id: {raw['matrix_id']}")
        if raw["matrix_id"] in matrix_ids:
            raise ComplianceRegistryError(f"duplicate matrix_id: {raw['matrix_id']}")
        matrix_ids.add(raw["matrix_id"])
        if not raw["venue"] or raw["venue"] != raw["venue"].lower():
            raise ComplianceRegistryError("venue must be nonempty lowercase")
        if not raw["jurisdiction"] or not raw["account_type"]:
            raise ComplianceRegistryError("jurisdiction and account_type are required")
        context = (raw["venue"], raw["jurisdiction"], raw["account_type"])
        if context in contexts:
            raise ComplianceRegistryError(f"duplicate context: {context}")
        contexts.add(context)
        reviews = (
            raw["eligibility_review"],
            raw["api_terms_review"],
            raw["trading_review"],
        )
        if any(review not in REVIEW_STATUSES for review in reviews):
            raise ComplianceRegistryError("invalid review status")
        if raw["status"] == "GREEN":
            if reviews != ("VERIFIED", "VERIFIED", "VERIFIED"):
                raise ComplianceRegistryError("GREEN requires all reviews VERIFIED")
            if "UNSPECIFIED" in context:
                raise ComplianceRegistryError("GREEN requires an exact operating context")
            if not _APPROVAL_REF.fullmatch(raw["approval_ref"]):
                raise ComplianceRegistryError("GREEN requires Team I approval_ref")
        elif raw["approval_ref"]:
            raise ComplianceRegistryError("non-GREEN rows cannot carry approval_ref")
        rows.append(ComplianceRow(**raw))
    return tuple(rows)


def load_data_license_register(root: str | Path) -> tuple[DataLicenseRow, ...]:
    """Load license reviews keyed only by existing stable catalog IDs."""

    raw_rows = _read_csv(root, "data_license_register.csv", LICENSE_COLUMNS)
    ids = [row["catalog_item_id"] for row in raw_rows]
    if tuple(ids) != EXPECTED_LICENSE_REVIEW_IDS:
        raise ComplianceRegistryError(
            "catalog_item_id rows must be exactly O-001 through O-008, "
            "R-039, I-018, R-042, and R-043 in order"
        )
    stable_catalog_ids = _stable_catalog_ids(root)
    missing_from_catalog = sorted(set(ids) - stable_catalog_ids)
    if missing_from_catalog:
        raise ComplianceRegistryError(
            "license review IDs are absent from the stable catalog: "
            + ", ".join(missing_from_catalog)
        )
    rows: list[DataLicenseRow] = []
    for raw in raw_rows:
        _validate_common(raw)
        if not raw["source_name"]:
            raise ComplianceRegistryError("source_name is required")
        if raw["commercial_use"] not in USE_STATUSES:
            raise ComplianceRegistryError("invalid commercial_use")
        if raw["redistribution"] not in USE_STATUSES:
            raise ComplianceRegistryError("invalid redistribution")
        if raw["attribution_required"] not in ATTRIBUTION_STATUSES:
            raise ComplianceRegistryError("invalid attribution_required")
        if raw["operational_use"] not in OPERATIONAL_USE_STATUSES:
            raise ComplianceRegistryError("invalid operational_use")
        if raw["status"] == "GREEN":
            if raw["commercial_use"] not in {
                "PERMITTED",
                "PERMITTED_WITH_CONDITIONS",
            }:
                raise ComplianceRegistryError(
                    "license GREEN requires permitted commercial_use"
                )
            if raw["redistribution"] not in {
                "PERMITTED",
                "PERMITTED_WITH_CONDITIONS",
            }:
                raise ComplianceRegistryError(
                    "license GREEN requires known permitted redistribution"
                )
            if raw["attribution_required"] == "UNKNOWN":
                raise ComplianceRegistryError(
                    "license GREEN requires known attribution requirement"
                )
            if raw["operational_use"] != "APPROVED":
                raise ComplianceRegistryError(
                    "license GREEN requires operational_use APPROVED"
                )
            if not _APPROVAL_REF.fullmatch(raw["approval_ref"]):
                raise ComplianceRegistryError(
                    "license GREEN requires Team I approval_ref"
                )
        elif raw["approval_ref"]:
            raise ComplianceRegistryError(
                "non-GREEN license rows cannot carry approval_ref"
            )
        rows.append(DataLicenseRow(**raw))
    return tuple(rows)


def may_execute_real_money(
    matrix: tuple[ComplianceRow, ...] | list[ComplianceRow],
    *,
    venue: str,
    jurisdiction: str,
    account_type: str,
) -> bool:
    """Return true only for one exact, fully verified Team I GREEN context."""

    matches = [
        row
        for row in matrix
        if (row.venue, row.jurisdiction, row.account_type)
        == (venue, jurisdiction, account_type)
    ]
    if len(matches) != 1:
        return False
    row = matches[0]
    context_is_green = (
        row.status == "GREEN"
        and not row.open_blocker
        and row.eligibility_review == "VERIFIED"
        and row.api_terms_review == "VERIFIED"
        and row.trading_review == "VERIFIED"
        and bool(_APPROVAL_REF.fullmatch(row.approval_ref))
        and "UNSPECIFIED" not in (row.venue, row.jurisdiction, row.account_type)
    )
    if not context_is_green:
        return False
    # Charter v0.2 lists real_money_execution in blocked_scope and the NO-GO
    # list.  Team I green is a necessary promotion gate, never sufficient
    # authorization.  Removing this hard stop requires a new governed program
    # decision, not a registry-row edit.
    return False
