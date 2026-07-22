"""Fail-closed experiment preregistration and result-reference validation."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_IDS = tuple(f"X-{number:02d}" for number in range(1, 11))
_EXPERIMENT_ID_SET = frozenset(EXPERIMENT_IDS)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ALLOWED_STATUSES = frozenset(
    {"registered", "running", "done", "failed", "abandoned"}
)


class ExperimentRegistryError(ValueError):
    """The registry is malformed, inconsistent, or has been altered."""


class UnregisteredExperimentError(ExperimentRegistryError):
    """A result refers to an experiment that was not preregistered."""


class InvalidResultReferenceError(ExperimentRegistryError):
    """A result reference is incomplete or malformed."""


class PreRegistrationEvaluationError(ExperimentRegistryError):
    """Evaluation began before the registration became effective."""


class UnauthorizedResultScopeError(ExperimentRegistryError):
    """The requested result scope is absent or not authorized."""


class UnresolvedDependencyError(ExperimentRegistryError):
    """A preregistered dependency has not completed."""


class UnresolvedRegistrationLockError(ExperimentRegistryError):
    """A scope depends on a registration choice that is still unlocked."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentRegistryError(
            "registration contains a non-canonical value"
        ) from exc
    return rendered.encode("utf-8")


def compute_registration_record_sha256(card: Mapping[str, Any]) -> str:
    """Hash the immutable base registration, excluding its hash and amendments."""

    base = {
        key: value
        for key, value in card.items()
        if key not in {"registration_record_sha256", "amendments"}
    }
    return "sha256:" + hashlib.sha256(_canonical_bytes(base)).hexdigest()


def compute_amendment_sha256(amendment: Mapping[str, Any]) -> str:
    content = {
        key: value for key, value in amendment.items() if key != "amendment_sha256"
    }
    return "sha256:" + hashlib.sha256(_canonical_bytes(content)).hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ExperimentRegistryError(f"missing CSV header: {path}")
            return list(reader.fieldnames), list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ExperimentRegistryError(f"cannot read registry CSV: {path}") from exc


def _require_string(value: Any, field: str, experiment_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentRegistryError(
            f"{experiment_id}: {field} must be a non-empty string"
        )
    return value


def _lock_map(card: Mapping[str, Any], experiment_id: str) -> dict[str, dict[str, Any]]:
    locks = card.get("registration_locks")
    if not isinstance(locks, list):
        raise ExperimentRegistryError(
            f"{experiment_id}: registration_locks must be a list"
        )
    result: dict[str, dict[str, Any]] = {}
    for index, lock in enumerate(locks):
        if not isinstance(lock, dict):
            raise ExperimentRegistryError(
                f"{experiment_id}: registration lock {index} must be an object"
            )
        lock_id = _require_string(lock.get("id"), "registration lock id", experiment_id)
        if lock_id in result:
            raise ExperimentRegistryError(
                f"{experiment_id}: duplicate registration lock {lock_id}"
            )
        if lock.get("status") not in {"unresolved", "resolved"}:
            raise ExperimentRegistryError(
                f"{experiment_id}: registration lock {lock_id} has invalid status"
            )
        _require_string(lock.get("reason"), "registration lock reason", experiment_id)
        result[lock_id] = lock
    return result


def _validate_scopes(
    card: Mapping[str, Any], experiment_id: str, locks: Mapping[str, Any]
) -> None:
    scopes = card.get("authorization_scopes")
    if not isinstance(scopes, dict) or not scopes:
        raise ExperimentRegistryError(
            f"{experiment_id}: authorization_scopes must be a non-empty object"
        )
    for scope_name, scope in scopes.items():
        _require_string(scope_name, "scope name", experiment_id)
        if not isinstance(scope, dict) or not isinstance(scope.get("authorized"), bool):
            raise ExperimentRegistryError(
                f"{experiment_id}: scope {scope_name} must declare authorized"
            )
        required = scope.get("required_lock_ids", [])
        if not isinstance(required, list) or not all(
            isinstance(item, str) and item for item in required
        ):
            raise ExperimentRegistryError(
                f"{experiment_id}: scope {scope_name} has malformed required_lock_ids"
            )
        unknown = sorted(set(required) - set(locks))
        if unknown:
            raise ExperimentRegistryError(
                f"{experiment_id}: scope {scope_name} references unknown registration lock: "
                + ", ".join(unknown)
            )
        if scope.get("permanent_no_go") is True and scope.get("authorized") is True:
            raise ExperimentRegistryError(
                f"{experiment_id}: permanent NO-GO scope {scope_name} is authorized"
            )
        label = scope.get("required_result_label")
        if label is not None and label not in {"PRELIMINARY", "FORMAL"}:
            raise ExperimentRegistryError(
                f"{experiment_id}: scope {scope_name} has invalid result label"
            )


def _validate_amendments(card: Mapping[str, Any], experiment_id: str) -> None:
    amendments = card.get("amendments", [])
    if not isinstance(amendments, list):
        raise ExperimentRegistryError(f"{experiment_id}: amendments must be a list")
    prior = card["registration_record_sha256"]
    for expected_sequence, amendment in enumerate(amendments, start=1):
        if not isinstance(amendment, dict):
            raise ExperimentRegistryError(
                f"{experiment_id}: amendment {expected_sequence} must be an object"
            )
        if amendment.get("sequence") != expected_sequence:
            raise ExperimentRegistryError(
                f"{experiment_id}: amendment sequence must be contiguous from 1"
            )
        if amendment.get("prior_sha256") != prior:
            raise ExperimentRegistryError(
                f"{experiment_id}: amendment {expected_sequence} prior hash mismatch"
            )
        _require_string(
            amendment.get("amended_at"), "amendment amended_at", experiment_id
        )
        _require_string(
            amendment.get("approved_by"), "amendment approved_by", experiment_id
        )
        _require_string(amendment.get("reason"), "amendment reason", experiment_id)
        if not isinstance(amendment.get("changes"), dict) or not amendment["changes"]:
            raise ExperimentRegistryError(
                f"{experiment_id}: amendment {expected_sequence} changes must be non-empty"
            )
        expected_hash = compute_amendment_sha256(amendment)
        if amendment.get("amendment_sha256") != expected_hash:
            raise ExperimentRegistryError(
                f"{experiment_id}: amendment {expected_sequence} content hash mismatch"
            )
        prior = expected_hash


def _validate_card(card: dict[str, Any], experiment_id: str) -> None:
    required_strings = (
        "name",
        "owner_team",
        "status",
        "hypothesis",
        "method",
        "split",
        "pass_criteria",
        "fail_criteria",
        "registered_at",
        "result_acceptance_not_before",
    )
    for field in required_strings:
        _require_string(card.get(field), field, experiment_id)
    if card.get("id") != experiment_id:
        raise ExperimentRegistryError(
            f"{experiment_id}: filename/card experiment id mismatch"
        )
    if card["status"] not in _ALLOWED_STATUSES:
        raise ExperimentRegistryError(f"{experiment_id}: invalid status")
    if card.get("due_gate") is not None:
        raise ExperimentRegistryError(
            f"{experiment_id}: due_gate is not an experiment deadline and must be null"
        )
    if not isinstance(card.get("execution_authorized"), bool):
        raise ExperimentRegistryError(
            f"{experiment_id}: execution_authorized must be boolean"
        )
    for field in ("leakage_checks", "metrics", "dependencies"):
        value = card.get(field)
        if not isinstance(value, list) or (
            field != "dependencies"
            and not all(isinstance(item, str) and item.strip() for item in value)
        ):
            raise ExperimentRegistryError(f"{experiment_id}: malformed {field}")
    data = card.get("data")
    if not isinstance(data, list) or not data:
        raise ExperimentRegistryError(f"{experiment_id}: data must be non-empty")
    for index, data_input in enumerate(data):
        if not isinstance(data_input, dict):
            raise ExperimentRegistryError(
                f"{experiment_id}: data input {index} must be an object"
            )
        for field in ("source", "version", "pit_basis"):
            _require_string(
                data_input.get(field), f"data[{index}].{field}", experiment_id
            )
    lineage = card.get("source_lineage")
    if not isinstance(lineage, dict):
        raise ExperimentRegistryError(f"{experiment_id}: missing source_lineage")
    if lineage.get("charter_file") != "charter/research_program_charter_v0.2.md":
        raise ExperimentRegistryError(f"{experiment_id}: wrong charter lineage")
    if not isinstance(lineage.get("charter_sections"), list) or not lineage[
        "charter_sections"
    ]:
        raise ExperimentRegistryError(f"{experiment_id}: missing charter sections")
    if not isinstance(lineage.get("catalog_item_ids"), list):
        raise ExperimentRegistryError(f"{experiment_id}: malformed catalog lineage")
    gates = card.get("linked_first_artifact_due_gates")
    if not isinstance(gates, list) or any(
        not isinstance(item, dict)
        or set(item) != {"catalog_item_id", "due_gate"}
        or not item["catalog_item_id"]
        or not item["due_gate"]
        for item in gates
    ):
        raise ExperimentRegistryError(
            f"{experiment_id}: malformed linked first-artifact due gates"
        )
    locks = _lock_map(card, experiment_id)
    _validate_scopes(card, experiment_id, locks)
    registration_hash = card.get("registration_record_sha256")
    if registration_hash != compute_registration_record_sha256(card):
        raise ExperimentRegistryError(
            f"{experiment_id}: immutable registration record SHA-256 mismatch"
        )
    _validate_amendments(card, experiment_id)


def _catalog_gates(program_root: Path) -> dict[str, list[dict[str, str]]]:
    _, rows = _read_csv(program_root / "charter" / "catalog_registry.csv")
    result = {experiment_id: [] for experiment_id in EXPERIMENT_IDS}
    seen_catalog_ids: set[str] = set()
    for row in rows:
        catalog_id = row.get("catalog_item_id", "")
        if not catalog_id or catalog_id in seen_catalog_ids:
            raise ExperimentRegistryError("catalog registry has invalid stable IDs")
        seen_catalog_ids.add(catalog_id)
        linked = [] if row.get("linked_experiments") == "-" else row.get(
            "linked_experiments", ""
        ).split(";")
        for experiment_id in linked:
            if experiment_id not in _EXPERIMENT_ID_SET:
                raise ExperimentRegistryError(
                    f"catalog {catalog_id} links unknown experiment {experiment_id}"
                )
            result[experiment_id].append(
                {"catalog_item_id": catalog_id, "due_gate": row.get("due_gate", "")}
            )
    return result


def _check_dependency_graph(registry: Mapping[str, Mapping[str, Any]]) -> None:
    for experiment_id, card in registry.items():
        dependencies = card["dependencies"]
        if not all(isinstance(item, str) for item in dependencies):
            raise ExperimentRegistryError(f"{experiment_id}: malformed dependencies")
        unknown = sorted(set(dependencies) - set(registry))
        if unknown:
            raise ExperimentRegistryError(
                f"{experiment_id}: unknown dependencies: {', '.join(unknown)}"
            )
        if experiment_id in dependencies:
            raise ExperimentRegistryError(f"{experiment_id}: self dependency")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(experiment_id: str) -> None:
        if experiment_id in visiting:
            raise ExperimentRegistryError("experiment dependency cycle detected")
        if experiment_id in visited:
            return
        visiting.add(experiment_id)
        for dependency in registry[experiment_id]["dependencies"]:
            visit(dependency)
        visiting.remove(experiment_id)
        visited.add(experiment_id)

    for experiment_id in registry:
        visit(experiment_id)


def load_experiment_registry(program_root: str | Path) -> dict[str, dict[str, Any]]:
    """Load and cross-check the immutable cards, CSV index, and catalog lineage."""

    root = Path(program_root)
    registry_path = root / "registries" / "experiment_registry.csv"
    fields, rows = _read_csv(registry_path)
    required_fields = {
        "experiment_id",
        "card_path",
        "owner_team",
        "status",
        "execution_authorized",
        "registered_at",
        "due_gate",
        "card_sha256",
    }
    if not required_fields.issubset(fields):
        raise ExperimentRegistryError("experiment registry CSV has missing columns")
    row_by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        experiment_id = row.get("experiment_id", "")
        if experiment_id in row_by_id:
            raise ExperimentRegistryError(
                f"duplicate experiment registry row: {experiment_id}"
            )
        row_by_id[experiment_id] = row
    if set(row_by_id) != _EXPERIMENT_ID_SET:
        raise ExperimentRegistryError("registry must contain exactly X-01 through X-10")

    card_dir = root / "registries" / "experiments"
    try:
        card_paths = sorted(card_dir.glob("*.yaml"))
    except OSError as exc:
        raise ExperimentRegistryError("cannot enumerate experiment cards") from exc
    if {path.stem for path in card_paths} != _EXPERIMENT_ID_SET:
        raise ExperimentRegistryError(
            "card filenames must contain exactly X-01 through X-10"
        )

    expected_gates = _catalog_gates(root)
    registry: dict[str, dict[str, Any]] = {}
    for experiment_id in EXPERIMENT_IDS:
        row = row_by_id[experiment_id]
        expected_relative = f"registries/experiments/{experiment_id}.yaml"
        if row["card_path"] != expected_relative:
            raise ExperimentRegistryError(
                f"{experiment_id}: filename/card/CSV path mismatch"
            )
        path = root / expected_relative
        try:
            raw = path.read_bytes()
            document = yaml.safe_load(raw.decode("utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ExperimentRegistryError(f"cannot read card {experiment_id}") from exc
        actual_file_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
        if row["card_sha256"] != actual_file_hash:
            raise ExperimentRegistryError(f"{experiment_id}: card SHA-256 mismatch")
        if not isinstance(document, dict) or set(document) != {"experiment_card"}:
            raise ExperimentRegistryError(f"{experiment_id}: invalid card document")
        card = document["experiment_card"]
        if not isinstance(card, dict):
            raise ExperimentRegistryError(f"{experiment_id}: invalid experiment_card")
        _validate_card(card, experiment_id)
        if row["owner_team"] != card["owner_team"]:
            raise ExperimentRegistryError(f"{experiment_id}: owner mismatch")
        if row["status"] != card["status"]:
            raise ExperimentRegistryError(f"{experiment_id}: status mismatch")
        if row["execution_authorized"] != str(
            card["execution_authorized"]
        ).lower():
            raise ExperimentRegistryError(f"{experiment_id}: authorization mismatch")
        if row["registered_at"] != card["registered_at"]:
            raise ExperimentRegistryError(f"{experiment_id}: registration date mismatch")
        if row["due_gate"] or card["due_gate"] is not None:
            raise ExperimentRegistryError(
                f"{experiment_id}: experiment due_gate must remain empty/null"
            )
        if card["linked_first_artifact_due_gates"] != expected_gates[experiment_id]:
            raise ExperimentRegistryError(
                f"{experiment_id}: catalog first-artifact gate lineage mismatch"
            )
        if card["source_lineage"]["catalog_item_ids"] != [
            item["catalog_item_id"] for item in expected_gates[experiment_id]
        ]:
            raise ExperimentRegistryError(
                f"{experiment_id}: stable catalog ID lineage mismatch"
            )
        registry[experiment_id] = card
    _check_dependency_graph(registry)
    return registry


def _parse_utc_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise InvalidResultReferenceError(f"{field} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidResultReferenceError(
            f"{field} must be an ISO-8601 UTC timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise InvalidResultReferenceError(f"{field} must be UTC")
    return parsed


def validate_result_ref(
    program_root: str | Path,
    experiment_id: str,
    result_ref: str | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a result against a preregistered experiment and authorized scope."""

    root = Path(program_root)
    if experiment_id not in _EXPERIMENT_ID_SET or not (
        root / "registries" / "experiment_registry.csv"
    ).is_file():
        raise UnregisteredExperimentError(
            f"experiment {experiment_id} has no preexisting registration"
        )
    registry = load_experiment_registry(root)
    if experiment_id not in registry:
        raise UnregisteredExperimentError(
            f"experiment {experiment_id} has no preexisting registration"
        )
    if not isinstance(result_ref, Mapping):
        raise InvalidResultReferenceError(
            "registered results require scope plus code/data/result SHA-256 references"
        )
    required = {
        "scope",
        "result_label",
        "evaluation_started_at",
        "code_sha256",
        "data_sha256",
        "result_sha256",
    }
    if set(result_ref) != required:
        raise InvalidResultReferenceError(
            "result reference must contain exactly scope, label, evaluation time, and three hashes"
        )
    for hash_field in ("code_sha256", "data_sha256", "result_sha256"):
        if not isinstance(result_ref[hash_field], str) or not _SHA256_RE.fullmatch(
            result_ref[hash_field]
        ):
            raise InvalidResultReferenceError(f"invalid {hash_field}")

    card = registry[experiment_id]
    scope_name = result_ref["scope"]
    scopes = card["authorization_scopes"]
    if not isinstance(scope_name, str) or scope_name not in scopes:
        raise UnauthorizedResultScopeError(
            f"{experiment_id}: unknown result scope {scope_name!r}"
        )
    scope = scopes[scope_name]
    if scope["authorized"] is not True or scope.get("permanent_no_go") is True:
        raise UnauthorizedResultScopeError(
            f"{experiment_id}: result scope {scope_name} is not authorized"
        )
    required_label = scope.get("required_result_label", "FORMAL")
    if result_ref["result_label"] != required_label:
        raise UnauthorizedResultScopeError(
            f"{experiment_id}: scope {scope_name} requires {required_label} label"
        )
    evaluation_started = _parse_utc_timestamp(
        result_ref["evaluation_started_at"], "evaluation_started_at"
    )
    not_before = _parse_utc_timestamp(
        card["result_acceptance_not_before"], "result_acceptance_not_before"
    )
    if evaluation_started < not_before:
        raise PreRegistrationEvaluationError(
            f"{experiment_id}: evaluation predates effective preregistration"
        )

    incomplete = [
        dependency
        for dependency in card["dependencies"]
        if registry[dependency]["status"] != "done"
    ]
    if incomplete:
        raise UnresolvedDependencyError(
            f"{experiment_id}: unresolved dependencies: {', '.join(incomplete)}"
        )
    locks = _lock_map(card, experiment_id)
    unresolved = [
        lock_id
        for lock_id in scope.get("required_lock_ids", [])
        if locks[lock_id]["status"] != "resolved"
    ]
    if unresolved:
        raise UnresolvedRegistrationLockError(
            f"{experiment_id}: unresolved registration locks: {', '.join(unresolved)}"
        )
    return dict(result_ref)


__all__ = [
    "ExperimentRegistryError",
    "InvalidResultReferenceError",
    "PreRegistrationEvaluationError",
    "UnauthorizedResultScopeError",
    "UnregisteredExperimentError",
    "UnresolvedDependencyError",
    "UnresolvedRegistrationLockError",
    "compute_amendment_sha256",
    "compute_registration_record_sha256",
    "load_experiment_registry",
    "validate_result_ref",
]
