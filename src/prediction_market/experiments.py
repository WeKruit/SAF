"""Strict, append-only experiment preregistration and result validation."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_IDS = tuple(f"X-{number:02d}" for number in range(1, 11))
_EXPERIMENT_ID_SET = frozenset(EXPERIMENT_IDS)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_RESULT_ACCEPTANCE_NOT_BEFORE = "2026-07-23T00:00:00Z"
_ALLOWED_STATUSES = frozenset(
    {"registered", "running", "done", "failed", "abandoned"}
)
_PROGRAM_NO_GOS = frozenset(
    {
        "real_money_execution",
        "maker_strategy_live",
        "exact_queue_fill_claim_from_pmxt_l2",
        "multi_venue_simultaneous_execution_arbitrage",
        "live_copy_trading",
        "llm_hot_path",
        "reinforcement_learning",
        "large_scale_microservices",
        "readme_return_strategy_selection",
        "unregistered_quick_backtest",
    }
)

# This mapping is the in-runtime trust anchor for the initial program registration.
# The append-only Git history is the external monotonic anchor for later ledger rows.
_TRUSTED_BASE_REGISTRATIONS = {
    "X-01": "sha256:5a435284fbb455732abb99fb76dd7d494fc8d46c2adf4e211a0e9919156e4475",
    "X-02": "sha256:52af85bc9220764e62ce97b14e48ea416186fde6d5ec1197d7f680111dc81335",
    "X-03": "sha256:007111874de286924a370bb8e2bcb6db4afc35852c91bd4308e3d4ba42dc1960",
    "X-04": "sha256:9633dc71617a1f6f59ebb2c840c4a71c6df80138a44de6acd900a2e428a4b334",
    "X-05": "sha256:adf2c9ef16e82ddc6f70d18c2993992d4c5cd49a83facabad5937561a1fa4bf4",
    "X-06": "sha256:351a31fe2f7f2caf07f681e6181482904f8c77687237b59012221e4072fd91c1",
    "X-07": "sha256:0702864167a20829bafb0e316905c8a6264dec48c47396e5265118966e61b556",
    "X-08": "sha256:c105e84ca90ba68ad259185670fed99e51af7740d159838cf51156e9e36153c4",
    "X-09": "sha256:39235a6bb75f827b1260421a9c75ec2142cb8309698f03cad5d4f641b98909e1",
    "X-10": "sha256:51358f3c7fe47cdc34286c1b7fb552c7f7df2ab08743056a74e245d5a965ddb0",
}

_COMMON_CARD_FIELDS = frozenset(
    {
        "id",
        "name",
        "owner_team",
        "status",
        "hypothesis",
        "data",
        "method",
        "leakage_checks",
        "split",
        "metrics",
        "pass_criteria",
        "fail_criteria",
        "cost_estimate",
        "dependencies",
        "registered_at",
        "result_acceptance_not_before",
        "due_gate",
        "results_ref",
        "promotion_decision",
        "execution_authorized",
        "authorization_scopes",
        "registration_locks",
        "measurement_exemption",
        "falsified_direction_is_valid_measurement",
        "linked_first_artifact_due_gates",
        "source_lineage",
        "program_no_go_restrictions",
        "amendments",
        "registration_record_sha256",
    }
)
_OPTIONAL_CARD_FIELDS = {
    "X-01": {"deterministic_replay_required_levels"},
    "X-02": set(),
    "X-03": set(),
    "X-04": set(),
    "X-05": {"artifact_dependencies", "midpoint_allowed"},
    "X-06": {"decision_gates"},
    "X-07": {"midpoint_allowed"},
    "X-08": {"prospective_observation", "unresolved_decision_band"},
    "X-09": {"artifact_dependencies", "deterministic_replay_required_levels", "signal"},
    "X-10": {"recall_denominator_registered"},
}
_EXPERIMENT_REGISTRY_FIELDS = [
    "experiment_id",
    "card_path",
    "owner_team",
    "status",
    "execution_authorized",
    "registered_at",
    "due_gate",
    "card_sha256",
]
_CATALOG_FIELDS = [
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
]
_LEDGER_FIELDS = [
    "experiment_id",
    "sequence",
    "record_sha256",
    "prior_sha256",
    "amended_at",
    "approved_by",
    "reason",
]
_RESULT_FIELDS = frozenset(
    {
        "scope",
        "result_label",
        "evaluation_started_at",
        "code_sha256",
        "data_sha256",
        "result_sha256",
        "registration_head_sha256",
    }
)


class ExperimentRegistryError(ValueError):
    """The registry is malformed, inconsistent, or has been altered."""


class UnregisteredExperimentError(ExperimentRegistryError):
    """A result refers to an experiment without a preregistration."""


class InvalidResultReferenceError(ExperimentRegistryError):
    """A result reference is incomplete or inconsistent with preregistration."""


class PreRegistrationEvaluationError(ExperimentRegistryError):
    """Evaluation began before its controlling registration."""


class UnauthorizedResultScopeError(ExperimentRegistryError):
    """The requested result scope is absent or not authorized."""


class UnresolvedDependencyError(ExperimentRegistryError):
    """A preregistered dependency lacks completed, validated evidence."""


class UnresolvedRegistrationLockError(ExperimentRegistryError):
    """A scope depends on a registration choice that remains unresolved."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if type(key) is not str:
            raise ExperimentRegistryError("YAML object keys must be canonical strings")
        if key in result:
            raise ExperimentRegistryError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _canonical_value(value: Any, path: str = "$") -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is list:
        return [_canonical_value(item, f"{path}[]") for item in value]
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ExperimentRegistryError(
                    f"non-canonical object key at {path}: keys must be strings"
                )
            result[key] = _canonical_value(item, f"{path}.{key}")
        return result
    raise ExperimentRegistryError(
        f"non-canonical value at {path}: {type(value).__name__}"
    )


def _canonical_bytes(value: Any) -> bytes:
    canonical = _canonical_value(value)
    rendered = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return rendered.encode("utf-8")


def compute_registration_record_sha256(card: dict[str, Any]) -> str:
    """Hash the immutable base card, excluding its digest and amendment suffix."""

    if type(card) is not dict:
        raise ExperimentRegistryError("registration must be a canonical plain object")
    base = {
        key: value
        for key, value in card.items()
        if key not in {"registration_record_sha256", "amendments"}
    }
    return "sha256:" + hashlib.sha256(_canonical_bytes(base)).hexdigest()


def compute_amendment_sha256(amendment: dict[str, Any]) -> str:
    if type(amendment) is not dict:
        raise ExperimentRegistryError("amendment must be a canonical plain object")
    content = {
        key: value for key, value in amendment.items() if key != "amendment_sha256"
    }
    return "sha256:" + hashlib.sha256(_canonical_bytes(content)).hexdigest()


def _strict_csv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_fields:
                raise ExperimentRegistryError(
                    f"unexpected CSV columns in {path}: {reader.fieldnames!r}"
                )
            rows: list[dict[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row or set(row) != set(expected_fields):
                    raise ExperimentRegistryError(
                        f"malformed CSV row {line_number} in {path}"
                    )
                if any(type(value) is not str for value in row.values()):
                    raise ExperimentRegistryError(
                        f"malformed CSV row {line_number} in {path}"
                    )
                rows.append(row)  # type: ignore[arg-type]
            return rows
    except ExperimentRegistryError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ExperimentRegistryError(f"cannot read registry CSV: {path}") from exc


def _safe_file(root: Path, relative: str, purpose: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ExperimentRegistryError(f"invalid {purpose} path")
    lexical = root / relative
    try:
        root_resolved = root.resolve(strict=True)
        current = root_resolved
        relative_parts = Path(relative).parts
        if ".." in relative_parts:
            raise ExperimentRegistryError(f"{purpose} path escape")
        for part in relative_parts:
            current = current / part
            if current.is_symlink():
                raise ExperimentRegistryError(f"{purpose} path is a symlink")
        resolved = lexical.resolve(strict=True)
    except ExperimentRegistryError:
        raise
    except OSError as exc:
        raise ExperimentRegistryError(f"missing {purpose}: {relative}") from exc
    if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
        raise ExperimentRegistryError(f"{purpose} path escape or non-file")
    return resolved


def _load_yaml_card(path: Path, experiment_id: str) -> dict[str, Any]:
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except ExperimentRegistryError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ExperimentRegistryError(f"cannot read card {experiment_id}") from exc
    if type(document) is not dict or set(document) != {"experiment_card"}:
        raise ExperimentRegistryError(f"{experiment_id}: invalid card document")
    card = document["experiment_card"]
    if type(card) is not dict:
        raise ExperimentRegistryError(f"{experiment_id}: invalid experiment_card")
    _canonical_value(card)
    return card


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ExperimentRegistryError(f"{label} has unexpected or missing keys")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ExperimentRegistryError(f"{label} must be a non-empty canonical string")
    return value


def _sha256(value: Any, label: str, result_error: bool = False) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        if result_error:
            raise InvalidResultReferenceError(f"invalid {label}")
        raise ExperimentRegistryError(f"invalid {label}")
    return value


def _canonical_utc(
    value: Any, label: str, *, result_error: bool = False
) -> datetime:
    error_type = InvalidResultReferenceError if result_error else ExperimentRegistryError
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        raise error_type(f"{label} must be a canonical UTC timestamp ending in Z")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise error_type(f"{label} must be a canonical UTC timestamp") from exc


def _lock_map(card: dict[str, Any], experiment_id: str) -> dict[str, dict[str, Any]]:
    locks = card["registration_locks"]
    if type(locks) is not list:
        raise ExperimentRegistryError(f"{experiment_id}: registration_locks must be a list")
    result: dict[str, dict[str, Any]] = {}
    for lock in locks:
        lock = _exact_keys(lock, {"id", "status", "reason"}, f"{experiment_id} lock")
        lock_id = _nonempty_string(lock["id"], f"{experiment_id} lock id")
        if lock_id in result:
            raise ExperimentRegistryError(f"{experiment_id}: duplicate lock {lock_id}")
        if lock["status"] not in {"unresolved", "resolved"}:
            raise ExperimentRegistryError(f"{experiment_id}: invalid lock status")
        _nonempty_string(lock["reason"], f"{experiment_id} lock reason")
        result[lock_id] = lock
    return result


def _validate_scopes(
    card: dict[str, Any], experiment_id: str, locks: dict[str, dict[str, Any]]
) -> None:
    scopes = card["authorization_scopes"]
    if type(scopes) is not dict or not scopes:
        raise ExperimentRegistryError(f"{experiment_id}: invalid authorization_scopes")
    for scope_name, raw_scope in scopes.items():
        _nonempty_string(scope_name, f"{experiment_id} scope")
        if type(raw_scope) is not dict:
            raise ExperimentRegistryError(f"{experiment_id}: invalid scope {scope_name}")
        allowed = {"authorized", "required_result_label", "required_lock_ids"}
        if "permanent_no_go" in raw_scope:
            allowed.add("permanent_no_go")
        scope = _exact_keys(raw_scope, allowed, f"{experiment_id} scope {scope_name}")
        if type(scope["authorized"]) is not bool:
            raise ExperimentRegistryError(f"{experiment_id}: scope authorization must be bool")
        if scope["required_result_label"] not in {"FORMAL", "PRELIMINARY"}:
            raise ExperimentRegistryError(
                f"{experiment_id}: scope {scope_name} has invalid required_result_label"
            )
        required = scope["required_lock_ids"]
        if type(required) is not list or any(type(item) is not str for item in required):
            raise ExperimentRegistryError(f"{experiment_id}: invalid scope locks")
        if len(required) != len(set(required)):
            raise ExperimentRegistryError(f"{experiment_id}: duplicate scope lock")
        unknown = sorted(set(required) - set(locks))
        if unknown:
            raise ExperimentRegistryError(
                f"{experiment_id}: scope {scope_name} references unknown registration lock: "
                + ", ".join(unknown)
            )
        permanent = scope.get("permanent_no_go", False)
        if type(permanent) is not bool:
            raise ExperimentRegistryError(
                f"{experiment_id}: permanent_no_go must be boolean"
            )
        if permanent and scope["authorized"]:
            raise ExperimentRegistryError(
                f"{experiment_id}: permanent NO-GO scope cannot be authorized"
            )


def _validate_card_structure(card: dict[str, Any], experiment_id: str) -> None:
    expected = _COMMON_CARD_FIELDS | _OPTIONAL_CARD_FIELDS[experiment_id]
    if set(card) != expected:
        raise ExperimentRegistryError(
            f"{experiment_id}: card has unexpected or missing keys"
        )
    if card["id"] != experiment_id:
        raise ExperimentRegistryError(f"{experiment_id}: filename/card id mismatch")
    for field_name in (
        "name",
        "owner_team",
        "hypothesis",
        "method",
        "split",
        "pass_criteria",
        "fail_criteria",
    ):
        _nonempty_string(card[field_name], f"{experiment_id}.{field_name}")
    if card["status"] not in _ALLOWED_STATUSES:
        raise ExperimentRegistryError(f"{experiment_id}: invalid status")
    if type(card["registered_at"]) is not str or _DATE_RE.fullmatch(card["registered_at"]) is None:
        raise ExperimentRegistryError(f"{experiment_id}: invalid registered_at")
    if card["registered_at"] != "2026-07-22":
        raise ExperimentRegistryError(f"{experiment_id}: immutable registration date changed")
    if card["result_acceptance_not_before"] != _RESULT_ACCEPTANCE_NOT_BEFORE:
        raise ExperimentRegistryError(f"{experiment_id}: immutable preregistration boundary changed")
    if card["due_gate"] is not None:
        raise ExperimentRegistryError(f"{experiment_id}: due_gate must be null")
    if type(card["execution_authorized"]) is not bool:
        raise ExperimentRegistryError(f"{experiment_id}: execution_authorized must be bool")
    if type(card["measurement_exemption"]) is not bool or type(
        card["falsified_direction_is_valid_measurement"]
    ) is not bool:
        raise ExperimentRegistryError(f"{experiment_id}: invalid measurement flags")
    if type(card["promotion_decision"]) is not str:
        raise ExperimentRegistryError(f"{experiment_id}: invalid promotion_decision")
    if type(card["results_ref"]) is not list or card["results_ref"]:
        raise ExperimentRegistryError(f"{experiment_id}: immutable base results_ref must be []")
    if type(card["data"]) is not list or not card["data"]:
        raise ExperimentRegistryError(f"{experiment_id}: data must be non-empty")
    for index, item in enumerate(card["data"]):
        item = _exact_keys(
            item, {"source", "version", "pit_basis"}, f"{experiment_id} data[{index}]"
        )
        for field_name in ("source", "version", "pit_basis"):
            _nonempty_string(item[field_name], f"{experiment_id} data[{index}].{field_name}")
    for field_name in ("leakage_checks", "metrics", "dependencies"):
        value = card[field_name]
        if type(value) is not list or any(type(item) is not str or not item for item in value):
            raise ExperimentRegistryError(f"{experiment_id}: malformed {field_name}")
        if len(value) != len(set(value)):
            raise ExperimentRegistryError(f"{experiment_id}: duplicate {field_name}")
    cost = _exact_keys(card["cost_estimate"], {"compute", "human_days"}, f"{experiment_id} cost")
    _nonempty_string(cost["compute"], f"{experiment_id} cost compute")
    _nonempty_string(cost["human_days"], f"{experiment_id} cost human_days")
    locks = _lock_map(card, experiment_id)
    _validate_scopes(card, experiment_id, locks)
    lineage = _exact_keys(
        card["source_lineage"],
        {"charter_file", "charter_sections", "catalog_item_ids"},
        f"{experiment_id} source_lineage",
    )
    if lineage["charter_file"] != "charter/research_program_charter_v0.2.md":
        raise ExperimentRegistryError(f"{experiment_id}: wrong charter lineage")
    if type(lineage["charter_sections"]) is not list or not lineage["charter_sections"]:
        raise ExperimentRegistryError(f"{experiment_id}: missing charter sections")
    if type(lineage["catalog_item_ids"]) is not list:
        raise ExperimentRegistryError(f"{experiment_id}: malformed catalog lineage")
    gates = card["linked_first_artifact_due_gates"]
    if type(gates) is not list:
        raise ExperimentRegistryError(f"{experiment_id}: malformed catalog gates")
    for gate in gates:
        gate = _exact_keys(gate, {"catalog_item_id", "due_gate"}, f"{experiment_id} gate")
        _nonempty_string(gate["catalog_item_id"], f"{experiment_id} catalog id")
        _nonempty_string(gate["due_gate"], f"{experiment_id} catalog due gate")
    if set(card["program_no_go_restrictions"]) != _PROGRAM_NO_GOS or len(
        card["program_no_go_restrictions"]
    ) != len(_PROGRAM_NO_GOS):
        raise ExperimentRegistryError(f"{experiment_id}: program NO-GO drift")
    if "artifact_dependencies" in card:
        dependencies = card["artifact_dependencies"]
        if type(dependencies) is not list or not dependencies:
            raise ExperimentRegistryError(f"{experiment_id}: invalid artifact dependencies")
        for dependency in dependencies:
            dependency = _exact_keys(
                dependency, {"path", "version", "sha256"}, f"{experiment_id} artifact dependency"
            )
            _nonempty_string(dependency["path"], f"{experiment_id} artifact path")
            _nonempty_string(dependency["version"], f"{experiment_id} artifact version")
            _sha256(dependency["sha256"], f"{experiment_id} artifact SHA-256")
    if "prospective_observation" in card:
        observation = _exact_keys(
            card["prospective_observation"],
            {"actual_elapsed_days", "fixtures_can_satisfy_elapsed_time"},
            f"{experiment_id} prospective observation",
        )
        if type(observation["actual_elapsed_days"]) is not int or type(
            observation["fixtures_can_satisfy_elapsed_time"]
        ) is not bool:
            raise ExperimentRegistryError(f"{experiment_id}: invalid prospective observation")
    if type(card["amendments"]) is not list:
        raise ExperimentRegistryError(f"{experiment_id}: amendments must be a list")


def _catalog_gates(root: Path) -> dict[str, list[dict[str, str]]]:
    path = _safe_file(root, "charter/catalog_registry.csv", "catalog registry")
    rows = _strict_csv(path, _CATALOG_FIELDS)
    result = {experiment_id: [] for experiment_id in EXPERIMENT_IDS}
    seen_ids: set[str] = set()
    for row in rows:
        catalog_id = row["catalog_item_id"]
        if not catalog_id or catalog_id in seen_ids:
            raise ExperimentRegistryError("catalog registry has duplicate stable IDs")
        seen_ids.add(catalog_id)
        raw_links = row["linked_experiments"]
        if raw_links == "-" or raw_links == "":
            links: list[str] = []
        else:
            if raw_links != raw_links.strip():
                raise ExperimentRegistryError(f"catalog {catalog_id} link has whitespace")
            links = raw_links.split(";")
            if any(link != link.strip() for link in links):
                raise ExperimentRegistryError(f"catalog {catalog_id} link has whitespace")
            if len(links) != len(set(links)):
                raise ExperimentRegistryError(f"catalog {catalog_id} has duplicate experiment link")
        for experiment_id in links:
            if experiment_id not in _EXPERIMENT_ID_SET:
                raise ExperimentRegistryError(
                    f"catalog {catalog_id} links unknown experiment {experiment_id}"
                )
            result[experiment_id].append(
                {"catalog_item_id": catalog_id, "due_gate": row["due_gate"]}
            )
    return result


def _check_dependency_graph(cards: dict[str, dict[str, Any]]) -> None:
    for experiment_id, card in cards.items():
        unknown = sorted(set(card["dependencies"]) - set(cards))
        if unknown:
            raise ExperimentRegistryError(f"{experiment_id}: unknown dependencies")
        if experiment_id in card["dependencies"]:
            raise ExperimentRegistryError(f"{experiment_id}: self dependency")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(experiment_id: str) -> None:
        if experiment_id in visiting:
            raise ExperimentRegistryError("experiment dependency cycle detected")
        if experiment_id in visited:
            return
        visiting.add(experiment_id)
        for dependency in cards[experiment_id]["dependencies"]:
            visit(dependency)
        visiting.remove(experiment_id)
        visited.add(experiment_id)

    for experiment_id in cards:
        visit(experiment_id)


def _validate_result_shape(value: Any) -> dict[str, str]:
    if type(value) is not dict:
        raise InvalidResultReferenceError("result_ref must be a plain dict")
    if set(value) != _RESULT_FIELDS:
        raise InvalidResultReferenceError("result_ref has unexpected or missing fields")
    snapshot: dict[str, str] = {}
    for field_name in _RESULT_FIELDS:
        field_value = value[field_name]
        if type(field_value) is not str:
            raise InvalidResultReferenceError(f"{field_name} must be a string")
        snapshot[field_name] = field_value
    if not snapshot["scope"]:
        raise InvalidResultReferenceError("scope must be non-empty")
    if snapshot["result_label"] not in {"FORMAL", "PRELIMINARY"}:
        raise InvalidResultReferenceError("result_label must be FORMAL or PRELIMINARY")
    _canonical_utc(snapshot["evaluation_started_at"], "evaluation_started_at", result_error=True)
    for field_name in (
        "code_sha256",
        "data_sha256",
        "result_sha256",
        "registration_head_sha256",
    ):
        _sha256(snapshot[field_name], field_name, result_error=True)
    return snapshot


def _validate_changes(changes: Any, experiment_id: str) -> dict[str, Any]:
    if type(changes) is not dict or not changes:
        raise ExperimentRegistryError(f"{experiment_id}: amendment changes must be non-empty")
    allowed = {
        "status",
        "resolve_locks",
        "authorize_scopes",
        "preregistered_inputs",
        "results_ref",
    }
    if not set(changes).issubset(allowed):
        raise ExperimentRegistryError(f"{experiment_id}: uncontrolled amendment changes")
    if "status" in changes and changes["status"] not in _ALLOWED_STATUSES:
        raise ExperimentRegistryError(f"{experiment_id}: invalid amended status")
    if "resolve_locks" in changes:
        items = changes["resolve_locks"]
        if type(items) is not list:
            raise ExperimentRegistryError(f"{experiment_id}: resolve_locks must be a list")
        seen: set[str] = set()
        for item in items:
            item = _exact_keys(item, {"lock_id", "evidence_ref"}, f"{experiment_id} resolved lock")
            lock_id = _nonempty_string(item["lock_id"], f"{experiment_id} resolved lock id")
            if lock_id in seen:
                raise ExperimentRegistryError(f"{experiment_id}: duplicate resolved lock")
            seen.add(lock_id)
            _sha256(item["evidence_ref"], f"{experiment_id} lock evidence")
    if "authorize_scopes" in changes:
        scopes = changes["authorize_scopes"]
        if type(scopes) is not list or any(type(item) is not str for item in scopes):
            raise ExperimentRegistryError(f"{experiment_id}: authorize_scopes must be strings")
        if len(scopes) != len(set(scopes)):
            raise ExperimentRegistryError(f"{experiment_id}: duplicate authorized scope")
    if "preregistered_inputs" in changes:
        inputs = changes["preregistered_inputs"]
        if type(inputs) is not list:
            raise ExperimentRegistryError(f"{experiment_id}: preregistered_inputs must be a list")
        seen_inputs: set[str] = set()
        for item in inputs:
            item = _exact_keys(
                item, {"scope", "code_sha256", "data_sha256"}, f"{experiment_id} preregistered input"
            )
            scope = _nonempty_string(item["scope"], f"{experiment_id} input scope")
            if scope in seen_inputs:
                raise ExperimentRegistryError(f"{experiment_id}: duplicate input scope")
            seen_inputs.add(scope)
            _sha256(item["code_sha256"], f"{experiment_id} code_sha256")
            _sha256(item["data_sha256"], f"{experiment_id} data_sha256")
    if "results_ref" in changes:
        try:
            _validate_result_shape(changes["results_ref"])
        except InvalidResultReferenceError as exc:
            raise ExperimentRegistryError(f"{experiment_id}: invalid appended results_ref: {exc}") from exc
    return changes


@dataclass
class _RegistrationMeta:
    head: str
    head_at: str = _RESULT_ACCEPTANCE_NOT_BEFORE
    preregistered_inputs: dict[str, dict[str, str]] = field(default_factory=dict)
    stored_results: list[dict[str, str]] = field(default_factory=list)


def _scope_and_locks(
    card: dict[str, Any], scope_name: str
) -> tuple[dict[str, Any], list[str]]:
    scopes = card["authorization_scopes"]
    if scope_name not in scopes:
        raise UnauthorizedResultScopeError(f"{card['id']}: unknown result scope {scope_name!r}")
    scope = scopes[scope_name]
    if scope["authorized"] is not True or scope.get("permanent_no_go", False) is True:
        raise UnauthorizedResultScopeError(f"{card['id']}: result scope {scope_name} is not authorized")
    lock_by_id = {lock["id"]: lock for lock in card["registration_locks"]}
    unresolved = [
        lock_id
        for lock_id in scope["required_lock_ids"]
        if lock_by_id[lock_id]["status"] != "resolved"
    ]
    return scope, unresolved


def _validate_result_against_state(
    card: dict[str, Any], meta: _RegistrationMeta, result: dict[str, str]
) -> None:
    scope, unresolved = _scope_and_locks(card, result["scope"])
    if result["result_label"] != scope["required_result_label"]:
        raise UnauthorizedResultScopeError(
            f"{card['id']}: scope {result['scope']} requires {scope['required_result_label']} label"
        )
    if result["registration_head_sha256"] != meta.head:
        raise InvalidResultReferenceError(f"{card['id']}: registration head mismatch")
    registered_inputs = meta.preregistered_inputs.get(result["scope"])
    if registered_inputs is None:
        raise InvalidResultReferenceError(
            f"{card['id']}: no preregistered inputs for scope {result['scope']}"
        )
    if result["code_sha256"] != registered_inputs["code_sha256"] or result[
        "data_sha256"
    ] != registered_inputs["data_sha256"]:
        raise InvalidResultReferenceError(
            f"{card['id']}: result inputs do not match preregistered hashes"
        )
    if unresolved:
        raise UnresolvedRegistrationLockError(
            f"{card['id']}: unresolved registration locks: {', '.join(unresolved)}"
        )
    evaluation_at = _canonical_utc(
        result["evaluation_started_at"], "evaluation_started_at", result_error=True
    )
    boundary = _canonical_utc(
        card["result_acceptance_not_before"], "result_acceptance_not_before", result_error=True
    )
    input_registered_at = _canonical_utc(
        registered_inputs["registered_at"], "input preregistration", result_error=True
    )
    if evaluation_at < boundary:
        raise PreRegistrationEvaluationError(
            f"{card['id']}: evaluation predates effective preregistration"
        )
    if evaluation_at <= input_registered_at:
        raise PreRegistrationEvaluationError(
            f"{card['id']}: evaluation must follow input preregistration amendment"
        )


def _apply_amendments(
    base_card: dict[str, Any], ledger_rows: list[dict[str, str]]
) -> tuple[dict[str, Any], _RegistrationMeta]:
    experiment_id = base_card["id"]
    for expected_sequence, amendment in enumerate(base_card["amendments"], start=1):
        if type(amendment) is not dict or amendment.get("sequence") != expected_sequence:
            raise ExperimentRegistryError(
                f"{experiment_id}: amendment sequence must be contiguous from 1"
            )
    expected_rows = 1 + len(base_card["amendments"])
    if len(ledger_rows) != expected_rows:
        raise ExperimentRegistryError(f"{experiment_id}: ledger and card chain length mismatch")
    seed = ledger_rows[0]
    if seed != {
        "experiment_id": experiment_id,
        "sequence": "0",
        "record_sha256": base_card["registration_record_sha256"],
        "prior_sha256": "",
        "amended_at": "",
        "approved_by": "",
        "reason": "",
    }:
        raise ExperimentRegistryError(f"{experiment_id}: ledger base record mismatch")
    effective = copy.deepcopy(base_card)
    meta = _RegistrationMeta(head=base_card["registration_record_sha256"])
    prior_time = _canonical_utc(_RESULT_ACCEPTANCE_NOT_BEFORE, "registration boundary")
    for expected_sequence, amendment in enumerate(base_card["amendments"], start=1):
        expected_keys = {
            "sequence",
            "amended_at",
            "prior_sha256",
            "approved_by",
            "reason",
            "changes",
            "amendment_sha256",
        }
        amendment = _exact_keys(amendment, expected_keys, f"{experiment_id} amendment")
        if amendment["sequence"] != expected_sequence:
            raise ExperimentRegistryError(f"{experiment_id}: amendment sequence must be contiguous from 1")
        if amendment["prior_sha256"] != meta.head:
            raise ExperimentRegistryError(f"{experiment_id}: amendment prior hash mismatch")
        if amendment["approved_by"] not in {"H", "Team H"}:
            raise ExperimentRegistryError(f"{experiment_id}: amendment approved_by must be H")
        _nonempty_string(amendment["reason"], f"{experiment_id} amendment reason")
        amended_time = _canonical_utc(amendment["amended_at"], "amended_at")
        if amended_time <= prior_time:
            raise ExperimentRegistryError(f"{experiment_id}: amendment times must be strictly monotonic")
        if amendment["amendment_sha256"] != compute_amendment_sha256(amendment):
            raise ExperimentRegistryError(f"{experiment_id}: amendment content hash mismatch")
        ledger = ledger_rows[expected_sequence]
        expected_ledger = {
            "experiment_id": experiment_id,
            "sequence": str(expected_sequence),
            "record_sha256": amendment["amendment_sha256"],
            "prior_sha256": amendment["prior_sha256"],
            "amended_at": amendment["amended_at"],
            "approved_by": amendment["approved_by"],
            "reason": amendment["reason"],
        }
        if ledger != expected_ledger:
            raise ExperimentRegistryError(f"{experiment_id}: ledger and card chain record mismatch")
        changes = _validate_changes(amendment["changes"], experiment_id)
        if "resolve_locks" in changes:
            lock_by_id = {lock["id"]: lock for lock in effective["registration_locks"]}
            for item in changes["resolve_locks"]:
                if item["lock_id"] not in lock_by_id:
                    raise ExperimentRegistryError(f"{experiment_id}: unknown resolved lock")
                lock = lock_by_id[item["lock_id"]]
                if lock["status"] == "resolved":
                    raise ExperimentRegistryError(f"{experiment_id}: lock already resolved")
                lock["status"] = "resolved"
                lock["evidence_ref"] = item["evidence_ref"]
        if "authorize_scopes" in changes:
            for scope_name in changes["authorize_scopes"]:
                if scope_name not in effective["authorization_scopes"]:
                    raise ExperimentRegistryError(f"{experiment_id}: unknown authorized scope")
                scope = effective["authorization_scopes"][scope_name]
                if scope.get("permanent_no_go", False):
                    raise ExperimentRegistryError(f"{experiment_id}: cannot authorize permanent NO-GO")
                scope["authorized"] = True
        if "preregistered_inputs" in changes:
            for item in changes["preregistered_inputs"]:
                if item["scope"] not in effective["authorization_scopes"]:
                    raise ExperimentRegistryError(f"{experiment_id}: unknown input scope")
                meta.preregistered_inputs[item["scope"]] = {
                    "code_sha256": item["code_sha256"],
                    "data_sha256": item["data_sha256"],
                    "registered_at": amendment["amended_at"],
                }
        if "results_ref" in changes:
            result = _validate_result_shape(changes["results_ref"])
            try:
                _validate_result_against_state(effective, meta, result)
            except ExperimentRegistryError as exc:
                raise ExperimentRegistryError(
                    f"{experiment_id}: appended result does not match evaluation head: {exc}"
                ) from exc
            result_time = _canonical_utc(result["evaluation_started_at"], "evaluation_started_at")
            if amended_time <= result_time:
                raise ExperimentRegistryError(f"{experiment_id}: result amendment must follow evaluation")
            effective["results_ref"].append(result)
            meta.stored_results.append(result)
        if "status" in changes:
            effective["status"] = changes["status"]
        meta.head = amendment["amendment_sha256"]
        meta.head_at = amendment["amended_at"]
        prior_time = amended_time
    effective["registration_head_sha256"] = meta.head
    effective["preregistered_inputs"] = copy.deepcopy(meta.preregistered_inputs)
    return effective, meta


def _dependency_ready(
    experiment_id: str,
    registry: dict[str, dict[str, Any]],
    metadata: dict[str, _RegistrationMeta],
    visiting: set[str] | None = None,
) -> bool:
    visiting = set() if visiting is None else visiting
    if experiment_id in visiting:
        return False
    visiting.add(experiment_id)
    card = registry[experiment_id]
    meta = metadata[experiment_id]
    ready = (
        card["status"] == "done"
        and bool(meta.stored_results)
        and all(
            _dependency_ready(dependency, registry, metadata, visiting)
            for dependency in card["dependencies"]
        )
    )
    visiting.remove(experiment_id)
    return ready


def _load_registry_internal(
    program_root: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, _RegistrationMeta]]:
    root = Path(program_root)
    registry_csv = _safe_file(root, "registries/experiment_registry.csv", "experiment registry")
    rows = _strict_csv(registry_csv, _EXPERIMENT_REGISTRY_FIELDS)
    row_by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        experiment_id = row["experiment_id"]
        if experiment_id in row_by_id:
            raise ExperimentRegistryError(f"duplicate experiment registry row: {experiment_id}")
        row_by_id[experiment_id] = row
    if set(row_by_id) != _EXPERIMENT_ID_SET:
        raise ExperimentRegistryError("registry must contain exactly X-01 through X-10")

    catalog_gates = _catalog_gates(root)
    base_cards: dict[str, dict[str, Any]] = {}
    for experiment_id in EXPERIMENT_IDS:
        row = row_by_id[experiment_id]
        relative = f"registries/experiments/{experiment_id}.yaml"
        if row["card_path"] != relative:
            raise ExperimentRegistryError(f"{experiment_id}: filename/card/CSV path mismatch")
        path = _safe_file(root, relative, "experiment card")
        raw = path.read_bytes()
        if row["card_sha256"] != "sha256:" + hashlib.sha256(raw).hexdigest():
            raise ExperimentRegistryError(f"{experiment_id}: card SHA-256 mismatch")
        card = _load_yaml_card(path, experiment_id)
        _validate_card_structure(card, experiment_id)
        if card["registration_record_sha256"] != compute_registration_record_sha256(card):
            raise ExperimentRegistryError(
                f"{experiment_id}: immutable registration record SHA-256 mismatch"
            )
        if row["owner_team"] != card["owner_team"] or row["status"] != card["status"]:
            raise ExperimentRegistryError(f"{experiment_id}: CSV/card metadata mismatch")
        if row["execution_authorized"] != str(card["execution_authorized"]).lower():
            raise ExperimentRegistryError(f"{experiment_id}: CSV/card authorization mismatch")
        if row["registered_at"] != card["registered_at"] or row["due_gate"]:
            raise ExperimentRegistryError(f"{experiment_id}: CSV/card registration mismatch")
        if card["linked_first_artifact_due_gates"] != catalog_gates[experiment_id]:
            raise ExperimentRegistryError(f"{experiment_id}: catalog first-artifact gate mismatch")
        if card["source_lineage"]["catalog_item_ids"] != [
            item["catalog_item_id"] for item in catalog_gates[experiment_id]
        ]:
            raise ExperimentRegistryError(f"{experiment_id}: stable catalog lineage mismatch")
        base_cards[experiment_id] = card

    # Semantic errors are surfaced before the trust-anchor error so malformed test
    # fixtures receive the most specific rejection.
    _check_dependency_graph(base_cards)
    for experiment_id, card in base_cards.items():
        if card["registration_record_sha256"] != _TRUSTED_BASE_REGISTRATIONS[experiment_id]:
            raise ExperimentRegistryError(
                f"{experiment_id}: trusted base registration SHA-256 mismatch"
            )
        for dependency in card.get("artifact_dependencies", []):
            artifact = _safe_file(root, dependency["path"], "artifact dependency")
            digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
            if digest != dependency["sha256"]:
                raise ExperimentRegistryError(f"{experiment_id}: artifact dependency SHA-256 mismatch")

    ledger_path = _safe_file(
        root, "registries/experiment_amendment_ledger.csv", "amendment ledger"
    )
    ledger = _strict_csv(ledger_path, _LEDGER_FIELDS)
    ledger_by_id = {experiment_id: [] for experiment_id in EXPERIMENT_IDS}
    for row in ledger:
        if row["experiment_id"] not in _EXPERIMENT_ID_SET:
            raise ExperimentRegistryError("amendment ledger has unknown experiment")
        ledger_by_id[row["experiment_id"]].append(row)

    registry: dict[str, dict[str, Any]] = {}
    metadata: dict[str, _RegistrationMeta] = {}
    for experiment_id in EXPERIMENT_IDS:
        effective, meta = _apply_amendments(base_cards[experiment_id], ledger_by_id[experiment_id])
        registry[experiment_id] = effective
        metadata[experiment_id] = meta
    for experiment_id, card in registry.items():
        if card["status"] == "done" and not _dependency_ready(
            experiment_id, registry, metadata
        ):
            raise ExperimentRegistryError(
                f"{experiment_id}: done status lacks validated result/dependency evidence"
            )
    return registry, metadata


def load_experiment_registry(program_root: str | Path) -> dict[str, dict[str, Any]]:
    registry, _ = _load_registry_internal(program_root)
    return registry


def validate_result_ref(
    program_root: str | Path,
    experiment_id: str,
    result_ref: Any,
) -> dict[str, str]:
    root = Path(program_root)
    if experiment_id not in _EXPERIMENT_ID_SET or not (
        root / "registries" / "experiment_registry.csv"
    ).is_file():
        raise UnregisteredExperimentError(
            f"experiment {experiment_id} has no preexisting registration"
        )
    result = _validate_result_shape(result_ref)
    registry, metadata = _load_registry_internal(root)
    card = registry[experiment_id]
    meta = metadata[experiment_id]
    _validate_result_against_state(card, meta, result)
    incomplete = [
        dependency
        for dependency in card["dependencies"]
        if not _dependency_ready(dependency, registry, metadata)
    ]
    if incomplete:
        raise UnresolvedDependencyError(
            f"{experiment_id}: unresolved dependencies: {', '.join(incomplete)}"
        )
    return dict(result)


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
