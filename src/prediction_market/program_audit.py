"""Program-level artifact and NO-GO audit for the first-round handoff."""

from __future__ import annotations

import ast
import csv
import io
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from prediction_market.compliance import (
    ComplianceRegistryError,
    load_compliance_matrix,
    load_data_license_register,
    may_execute_real_money,
)


_ARTIFACT_COLUMNS = (
    "artifact_id",
    "path",
    "owner_team",
    "version",
    "due_gate",
    "status",
)
_DATASET_COLUMNS = (
    "dataset_id",
    "name",
    "use_class",
    "catalog_item_ids",
    "canonical_url",
    "source_version",
    "coverage",
    "grain",
    "auth",
    "license",
    "license_status",
    "license_review_id",
    "timestamp_semantics",
    "allowed_experiments",
    "manifest_sha256",
    "status",
    "owner",
    "version",
    "due_gate",
)
_MODEL_COLUMNS = (
    "model_id",
    "model_version",
    "source_catalog_item_ids",
    "experiment_id",
    "target",
    "horizon",
    "state_space",
    "pit_feature_contract",
    "data_manifest_sha256",
    "training_manifest_sha256",
    "parameter_config_sha256",
    "seed",
    "metrics",
    "status",
    "owner",
    "due_gate",
)
_EXPERIMENT_REGISTRY_COLUMNS = (
    "experiment_id",
    "card_path",
    "owner_team",
    "status",
    "execution_authorized",
    "registered_at",
    "due_gate",
    "card_sha256",
)
_ARTIFACT_ID = re.compile(r"ART-[A-Z0-9][A-Z0-9-]*")
_DATASET_ID = re.compile(r"DS-[A-Z0-9][A-Z0-9-]*")
_MODEL_ID = re.compile(r"MODEL-[A-Z0-9][A-Z0-9-]*")
_CATALOG_ID = re.compile(r"(?:R|I|O)-[0-9]{3}")
_EXPERIMENT_ID = re.compile(r"X-[0-9]{2,}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_VERSION = re.compile(r"v[0-9]+(?:\.[0-9]+)?")
_ALLOWED_STATUS = frozenset(
    {
        "registered",
        "complete",
        "blocked",
        "in_progress",
        "harness_pass",
        "PRELIMINARY_RESEARCH_ONLY",
        "RETROSPECTIVE_RESEARCH_ONLY",
        "POC_ONLY",
    }
)


class ProgramAuditError(ValueError):
    """A program artifact registry cannot be trusted."""


class ResearchRegistryError(ProgramAuditError):
    """A dataset or model registry is malformed or has a broken foreign key."""


class FormalResearchInputError(ResearchRegistryError):
    """A formal result uses untrusted or incompletely registered research inputs."""


@dataclass(frozen=True, slots=True)
class ArtifactRegistryRow:
    artifact_id: str
    path: str
    owner_team: str
    version: str
    due_gate: str
    status: str


@dataclass(frozen=True, slots=True)
class DatasetRegistryRow:
    dataset_id: str
    name: str
    use_class: str
    catalog_item_ids: tuple[str, ...]
    canonical_url: str
    source_version: str
    coverage: str
    grain: str
    auth: str
    license: str
    license_status: str
    license_review_id: str
    timestamp_semantics: str
    allowed_experiments: tuple[str, ...]
    manifest_sha256: str
    status: str
    owner: str
    version: str
    due_gate: str


@dataclass(frozen=True, slots=True)
class ModelRegistryRow:
    model_id: str
    model_version: str
    source_catalog_item_ids: tuple[str, ...]
    experiment_id: str
    target: str
    horizon: str
    state_space: tuple[str, ...]
    pit_feature_contract: str
    data_manifest_sha256: str
    training_manifest_sha256: str
    parameter_config_sha256: str
    seed: str
    metrics: tuple[str, ...]
    status: str
    owner: str
    due_gate: str


@dataclass(frozen=True, slots=True)
class NoGoAuditReport:
    violations: tuple[str, ...]
    real_money_authorized: bool
    live_maker_present: bool
    live_arbitrage_present: bool
    llm_hot_path_present: bool


def _safe_artifact(root: Path, relative_text: str) -> Path:
    if not relative_text or "\\" in relative_text:
        raise ProgramAuditError("artifact path must be a canonical relative POSIX path")
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ProgramAuditError("artifact path escapes program root")
    path = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ProgramAuditError(f"artifact path traverses symlink: {relative_text}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ProgramAuditError(f"artifact path is missing or unsafe: {relative_text}") from error
    if not resolved.is_file():
        raise ProgramAuditError(f"artifact path is not a file: {relative_text}")
    return resolved


def load_artifact_registry(root: str | Path) -> tuple[ArtifactRegistryRow, ...]:
    program_root = Path(root).resolve()
    path = _safe_artifact(program_root, "registries/artifact_registry.csv")
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ProgramAuditError("artifact registry must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if tuple(reader.fieldnames or ()) != _ARTIFACT_COLUMNS:
        raise ProgramAuditError("artifact registry columns do not match v0")
    rows: list[ArtifactRegistryRow] = []
    ids: set[str] = set()
    paths: set[str] = set()
    try:
        for raw in reader:
            if None in raw or any(
                raw[field] is None
                or not raw[field]
                or raw[field] != raw[field].strip()
                for field in _ARTIFACT_COLUMNS
            ):
                raise ProgramAuditError("artifact registry has empty or non-canonical cells")
            if not _ARTIFACT_ID.fullmatch(raw["artifact_id"]):
                raise ProgramAuditError(f"invalid artifact_id: {raw['artifact_id']}")
            if raw["artifact_id"] in ids:
                raise ProgramAuditError(f"duplicate artifact_id: {raw['artifact_id']}")
            if raw["path"] in paths:
                raise ProgramAuditError(f"duplicate artifact path: {raw['path']}")
            ids.add(raw["artifact_id"])
            paths.add(raw["path"])
            if not _VERSION.fullmatch(raw["version"]):
                raise ProgramAuditError(f"invalid artifact version: {raw['version']}")
            if raw["status"] not in _ALLOWED_STATUS:
                raise ProgramAuditError(f"invalid artifact status: {raw['status']}")
            _safe_artifact(program_root, raw["path"])
            rows.append(ArtifactRegistryRow(**raw))
    except csv.Error as error:
        raise ProgramAuditError(f"invalid artifact registry CSV: {error}") from error
    if not rows:
        raise ProgramAuditError("artifact registry must not be empty")
    return tuple(rows)


def _strict_registry_rows(
    root: Path,
    relative_path: str,
    columns: tuple[str, ...],
    *,
    allow_empty_fields: frozenset[str] = frozenset(),
) -> list[dict[str, str]]:
    path = _safe_artifact(root, relative_path)
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ResearchRegistryError(
            f"{relative_path} must be readable UTF-8"
        ) from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if tuple(reader.fieldnames or ()) != columns:
        raise ResearchRegistryError(f"{relative_path} columns do not match contract")
    rows: list[dict[str, str]] = []
    try:
        for raw in reader:
            if None in raw or set(raw) != set(columns):
                raise ResearchRegistryError(f"malformed row in {relative_path}")
            row: dict[str, str] = {}
            for field in columns:
                value = raw[field]
                if (
                    value is None
                    or value != value.strip()
                    or (not value and field not in allow_empty_fields)
                ):
                    raise ResearchRegistryError(
                        f"{relative_path} has empty or non-canonical cells"
                    )
                row[field] = value
            rows.append(row)
    except csv.Error as error:
        raise ResearchRegistryError(f"invalid registry CSV: {relative_path}") from error
    if not rows:
        raise ResearchRegistryError(f"{relative_path} must not be empty")
    return rows


def _split_registry_values(
    value: str, separator: str, label: str
) -> tuple[str, ...]:
    if value == "INVENTORY_ONLY":
        return ()
    values = tuple(value.split(separator))
    if (
        not values
        or any(not item or item != item.strip() for item in values)
        or len(values) != len(set(values))
    ):
        raise ResearchRegistryError(f"invalid or duplicate {label}")
    return values


def _catalog_ids(root: Path) -> set[str]:
    rows = _strict_registry_rows(
        root,
        "charter/catalog_registry.csv",
        (
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
        ),
        allow_empty_fields=frozenset({"secondary_teams"}),
    )
    return {row["catalog_item_id"] for row in rows}


def _registered_experiment_ids(root: Path) -> set[str]:
    path = _safe_artifact(root, "registries/experiment_registry.csv")
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ResearchRegistryError(
            "experiment registry must be readable UTF-8"
        ) from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if tuple(reader.fieldnames or ()) != _EXPERIMENT_REGISTRY_COLUMNS:
        raise ResearchRegistryError("experiment registry columns do not match contract")
    identifiers: set[str] = set()
    try:
        for raw in reader:
            if (
                None in raw
                or set(raw) != set(_EXPERIMENT_REGISTRY_COLUMNS)
                or any(type(value) is not str for value in raw.values())
            ):
                raise ResearchRegistryError("experiment registry has a malformed row")
            experiment_id = raw.get("experiment_id")
            if (
                type(experiment_id) is not str
                or not _EXPERIMENT_ID.fullmatch(experiment_id)
                or experiment_id in identifiers
            ):
                raise ResearchRegistryError(
                    "experiment registry has invalid or duplicate experiment_id"
                )
            identifiers.add(experiment_id)
    except csv.Error as error:
        raise ResearchRegistryError("invalid experiment registry CSV") from error
    expected = {f"X-{number:02d}" for number in range(1, 13)}
    if identifiers != expected:
        raise ResearchRegistryError(
            "experiment registry must contain exactly X-01 through X-12"
        )
    return identifiers


def load_dataset_registry(root: str | Path) -> tuple[DatasetRegistryRow, ...]:
    program_root = Path(root).resolve()
    raw_rows = _strict_registry_rows(
        program_root, "registries/dataset_registry.csv", _DATASET_COLUMNS
    )
    known_catalog_ids = _catalog_ids(program_root)
    known_experiment_ids = _registered_experiment_ids(program_root)
    try:
        license_reviews = {
            row.catalog_item_id: row
            for row in load_data_license_register(program_root)
        }
    except ComplianceRegistryError as error:
        raise ResearchRegistryError(
            f"data license register is invalid: {error}"
        ) from error
    rows: list[DatasetRegistryRow] = []
    seen: set[str] = set()
    for raw in raw_rows:
        dataset_id = raw["dataset_id"]
        if not _DATASET_ID.fullmatch(dataset_id) or dataset_id in seen:
            raise ResearchRegistryError(f"invalid or duplicate dataset_id: {dataset_id}")
        seen.add(dataset_id)
        catalog_ids = _split_registry_values(
            raw["catalog_item_ids"], ";", "dataset catalog_item_ids"
        )
        if any(
            not _CATALOG_ID.fullmatch(item) or item not in known_catalog_ids
            for item in catalog_ids
        ):
            raise ResearchRegistryError(f"{dataset_id}: unknown catalog foreign key")
        experiments = _split_registry_values(
            raw["allowed_experiments"], ";", "allowed_experiments"
        )
        if any(
            not _EXPERIMENT_ID.fullmatch(item) or item not in known_experiment_ids
            for item in experiments
        ):
            raise ResearchRegistryError(
                f"{dataset_id}: unknown experiment foreign key"
            )
        if raw["use_class"] not in {"canonical", "secondary", "blocked"}:
            raise ResearchRegistryError(f"{dataset_id}: invalid use_class")
        if raw["license_status"] not in {
            "approved",
            "research_only",
            "pending",
            "unknown",
            "blocked",
        }:
            raise ResearchRegistryError(f"{dataset_id}: invalid license_status")
        license_review_id = raw["license_review_id"]
        review = license_reviews.get(license_review_id)
        if (
            not _CATALOG_ID.fullmatch(license_review_id)
            or license_review_id not in known_catalog_ids
            or review is None
        ):
            raise ResearchRegistryError(
                f"{dataset_id}: unknown license review foreign key "
                f"{license_review_id}"
            )
        if raw["license_status"] == "approved" and not (
            review.status == "GREEN"
            and review.operational_use == "APPROVED"
            and bool(review.approval_ref)
        ):
            raise ResearchRegistryError(
                f"{dataset_id}: approved license requires {license_review_id} "
                "to be GREEN with operational approval"
            )
        if raw["status"] not in {"registered", "blocked"}:
            raise ResearchRegistryError(f"{dataset_id}: invalid status")
        if (raw["use_class"] == "blocked") != (raw["status"] == "blocked"):
            raise ResearchRegistryError(
                f"{dataset_id}: blocked use_class and status must agree"
            )
        if not raw["canonical_url"].startswith("https://"):
            raise ResearchRegistryError(f"{dataset_id}: canonical_url must use HTTPS")
        manifest = raw["manifest_sha256"]
        if manifest != "UNRESOLVED" and not _SHA256.fullmatch(manifest):
            raise ResearchRegistryError(f"{dataset_id}: invalid manifest_sha256")
        if not _VERSION.fullmatch(raw["version"]):
            raise ResearchRegistryError(f"{dataset_id}: invalid registry version")
        rows.append(
            DatasetRegistryRow(
                dataset_id=dataset_id,
                name=raw["name"],
                use_class=raw["use_class"],
                catalog_item_ids=catalog_ids,
                canonical_url=raw["canonical_url"],
                source_version=raw["source_version"],
                coverage=raw["coverage"],
                grain=raw["grain"],
                auth=raw["auth"],
                license=raw["license"],
                license_status=raw["license_status"],
                license_review_id=license_review_id,
                timestamp_semantics=raw["timestamp_semantics"],
                allowed_experiments=experiments,
                manifest_sha256=manifest,
                status=raw["status"],
                owner=raw["owner"],
                version=raw["version"],
                due_gate=raw["due_gate"],
            )
        )
    return tuple(rows)


def load_model_registry(root: str | Path) -> tuple[ModelRegistryRow, ...]:
    program_root = Path(root).resolve()
    raw_rows = _strict_registry_rows(
        program_root, "registries/model_registry.csv", _MODEL_COLUMNS
    )
    known_catalog_ids = _catalog_ids(program_root)
    known_experiment_ids = _registered_experiment_ids(program_root)
    rows: list[ModelRegistryRow] = []
    seen: set[str] = set()
    for raw in raw_rows:
        model_id = raw["model_id"]
        if not _MODEL_ID.fullmatch(model_id) or model_id in seen:
            raise ResearchRegistryError(f"invalid or duplicate model_id: {model_id}")
        seen.add(model_id)
        catalog_ids = _split_registry_values(
            raw["source_catalog_item_ids"], ";", "model catalog_item_ids"
        )
        if any(
            not _CATALOG_ID.fullmatch(item) or item not in known_catalog_ids
            for item in catalog_ids
        ):
            raise ResearchRegistryError(f"{model_id}: unknown catalog foreign key")
        experiment_id = raw["experiment_id"]
        if (
            not _EXPERIMENT_ID.fullmatch(experiment_id)
            or experiment_id not in known_experiment_ids
        ):
            raise ResearchRegistryError(f"{model_id}: unknown experiment foreign key")
        state_space = _split_registry_values(raw["state_space"], "|", "state_space")
        if len(state_space) < 2:
            raise ResearchRegistryError(f"{model_id}: state_space must have multiple states")
        if raw["horizon"] not in {"game_end", "next_state_transition"}:
            raise ResearchRegistryError(f"{model_id}: invalid horizon")
        if raw["model_version"] not in {"v0", "v1"}:
            raise ResearchRegistryError(f"{model_id}: invalid model_version")
        if raw["horizon"] == "next_state_transition" and raw["model_version"] != "v1":
            raise ResearchRegistryError(f"{model_id}: transition output must use v1")
        for field in (
            "data_manifest_sha256",
            "training_manifest_sha256",
            "parameter_config_sha256",
        ):
            value = raw[field]
            if value != "UNRESOLVED" and not _SHA256.fullmatch(value):
                raise ResearchRegistryError(f"{model_id}: invalid {field}")
        pit_contract = raw["pit_feature_contract"]
        if not (
            pit_contract.startswith("unresolved:")
            or _SHA256.fullmatch(pit_contract)
        ):
            raise ResearchRegistryError(f"{model_id}: invalid pit_feature_contract")
        if raw["status"] not in {"registered", "blocked", "poc_only"}:
            raise ResearchRegistryError(f"{model_id}: invalid status")
        if raw["seed"] != "UNRESOLVED" and not raw["seed"].isdigit():
            raise ResearchRegistryError(f"{model_id}: invalid seed")
        metrics = _split_registry_values(raw["metrics"], "|", "metrics")
        rows.append(
            ModelRegistryRow(
                model_id=model_id,
                model_version=raw["model_version"],
                source_catalog_item_ids=catalog_ids,
                experiment_id=experiment_id,
                target=raw["target"],
                horizon=raw["horizon"],
                state_space=state_space,
                pit_feature_contract=pit_contract,
                data_manifest_sha256=raw["data_manifest_sha256"],
                training_manifest_sha256=raw["training_manifest_sha256"],
                parameter_config_sha256=raw["parameter_config_sha256"],
                seed=raw["seed"],
                metrics=metrics,
                status=raw["status"],
                owner=raw["owner"],
                due_gate=raw["due_gate"],
            )
        )
    return tuple(rows)


def validate_registered_research_bindings(
    root: str | Path,
    *,
    experiment_id: str,
    dataset_ids: list[str] | tuple[str, ...],
    model_ids: list[str] | tuple[str, ...],
    result_class: str,
) -> tuple[tuple[DatasetRegistryRow, ...], tuple[ModelRegistryRow, ...]]:
    """Validate exact registered data/model eligibility without loading experiment cards."""

    if result_class not in {"formal", "poc"}:
        raise FormalResearchInputError("result_class must be formal or poc")

    program_root = Path(root).resolve()
    if experiment_id not in _registered_experiment_ids(program_root):
        raise FormalResearchInputError(f"experiment {experiment_id} is not registered")
    datasets_by_id = {row.dataset_id: row for row in load_dataset_registry(program_root)}
    models_by_id = {row.model_id: row for row in load_model_registry(program_root)}
    selected_datasets: list[DatasetRegistryRow] = []
    for dataset_id in dataset_ids:
        row = datasets_by_id.get(dataset_id)
        if row is None:
            raise FormalResearchInputError(f"dataset {dataset_id} is not registered")
        allowed_licenses = (
            {"approved"} if result_class == "formal" else {"approved", "research_only"}
        )
        if row.status == "blocked" or row.license_status not in allowed_licenses:
            raise FormalResearchInputError(
                f"dataset {dataset_id} license/status is blocked for {result_class} use"
            )
        if row.manifest_sha256 == "UNRESOLVED":
            raise FormalResearchInputError(
                f"dataset {dataset_id} has no resolved manifest hash"
            )
        if experiment_id not in row.allowed_experiments:
            raise FormalResearchInputError(
                f"dataset {dataset_id} is not allowed for experiment {experiment_id}"
            )
        selected_datasets.append(row)
    selected_models: list[ModelRegistryRow] = []
    for model_id in model_ids:
        row = models_by_id.get(model_id)
        if row is None:
            raise FormalResearchInputError(f"model {model_id} is not registered")
        allowed_model_statuses = (
            {"registered"}
            if result_class == "formal"
            else {"registered", "poc_only"}
        )
        if (
            row.experiment_id != experiment_id
            or row.status not in allowed_model_statuses
        ):
            raise FormalResearchInputError(
                f"model {model_id} is not authorized for {result_class} experiment use"
            )
        missing = [
            field
            for field, value in (
                ("PIT feature contract", row.pit_feature_contract),
                ("data manifest", row.data_manifest_sha256),
                ("training manifest", row.training_manifest_sha256),
                ("parameter/config hash", row.parameter_config_sha256),
            )
            if not _SHA256.fullmatch(value)
        ]
        if not row.seed.isdigit():
            missing.append("seed")
        if missing:
            raise FormalResearchInputError(
                f"model {model_id} lacks " + ", ".join(missing)
            )
        selected_models.append(row)
    return tuple(selected_datasets), tuple(selected_models)


def validate_research_inputs(
    root: str | Path,
    *,
    experiment_id: str,
    scope_name: str,
    dataset_ids: list[str] | tuple[str, ...],
    model_ids: list[str] | tuple[str, ...],
) -> tuple[tuple[DatasetRegistryRow, ...], tuple[ModelRegistryRow, ...]]:
    """Use the same card binding and eligibility rules as result acceptance."""

    from prediction_market.experiments import (
        load_experiment_registry,
        require_execution_authorized,
    )

    program_root = Path(root).resolve()
    experiments = load_experiment_registry(program_root)
    card = experiments.get(experiment_id)
    if card is None:
        raise FormalResearchInputError(f"experiment {experiment_id} is not registered")
    try:
        require_execution_authorized(card)
    except ValueError as error:
        raise FormalResearchInputError(str(error)) from error
    scope = card["authorization_scopes"].get(scope_name)
    if scope is None or scope["authorized"] is not True:
        raise FormalResearchInputError(
            f"experiment {experiment_id} scope {scope_name} is not authorized"
        )
    binding = scope.get("input_binding")
    if not isinstance(binding, dict) or binding.get("result_class") == "synthetic":
        raise FormalResearchInputError(
            f"experiment {experiment_id} scope {scope_name} has no research binding"
        )
    if list(dataset_ids) != binding["dataset_ids"] or list(model_ids) != binding[
        "model_ids"
    ]:
        raise FormalResearchInputError(
            f"experiment {experiment_id} scope {scope_name} input binding mismatch"
        )
    lock_by_id = {lock["id"]: lock for lock in card["registration_locks"]}
    unresolved = [
        lock_id
        for lock_id in scope["required_lock_ids"]
        if lock_by_id[lock_id]["status"] != "resolved"
    ]
    if unresolved:
        raise FormalResearchInputError(
            f"experiment {experiment_id} has unresolved required locks: "
            + ", ".join(unresolved)
        )
    return validate_registered_research_bindings(
        program_root,
        experiment_id=experiment_id,
        dataset_ids=dataset_ids,
        model_ids=model_ids,
        result_class=binding["result_class"],
    )


def validate_formal_research_inputs(
    root: str | Path,
    *,
    experiment_id: str,
    dataset_ids: list[str] | tuple[str, ...],
    model_ids: list[str] | tuple[str, ...],
) -> tuple[tuple[DatasetRegistryRow, ...], tuple[ModelRegistryRow, ...]]:
    """Fail closed unless every formal source and model is fully registered."""

    return validate_research_inputs(
        root,
        experiment_id=experiment_id,
        scope_name="formal_result",
        dataset_ids=dataset_ids,
        model_ids=model_ids,
    )


def audit_no_go(root: str | Path) -> NoGoAuditReport:
    """Verify executable surfaces remain inside Charter v0.2 blocked scope."""

    from prediction_market.experiments import load_experiment_registry

    program_root = Path(root).resolve()
    violations: list[str] = []
    matrix = load_compliance_matrix(program_root)
    real_money_authorized = any(
        may_execute_real_money(
            matrix,
            venue=row.venue,
            jurisdiction=row.jurisdiction,
            account_type=row.account_type,
        )
        for row in matrix
    )
    if real_money_authorized:
        violations.append("real-money execution is authorized")

    source_root = program_root / "src" / "prediction_market"
    python_files = tuple(source_root.rglob("*.py"))
    stems = {path.stem for path in python_files}
    live_maker_present = bool(stems & {"maker", "maker_live", "live_maker"})
    live_arbitrage_present = bool(
        stems & {"live_arbitrage", "multi_venue_arbitrage", "smart_order_router"}
    )
    copy_live_present = bool(stems & {"copy_trading", "live_copy_trading"})
    rl_present = bool(stems & {"rl", "reinforcement_learning"})
    llm_modules = {"openai", "anthropic", "langchain"}
    llm_hot_path_present = False
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = (node.module.split(".", 1)[0],)
            if llm_modules.intersection(imported):
                llm_hot_path_present = True
                break
        if llm_hot_path_present:
            break
    if live_maker_present:
        violations.append("live maker module is present")
    if live_arbitrage_present:
        violations.append("live multi-venue arbitrage module is present")
    if copy_live_present:
        violations.append("live copy-trading module is present")
    if rl_present:
        violations.append("reinforcement-learning module is present")
    if llm_hot_path_present:
        violations.append("LLM hot-path dependency is present")

    project = tomllib.loads((program_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = "\n".join(project["project"].get("dependencies", [])).lower()
    for forbidden in ("ray[rllib]", "stable-baselines", "fastapi", "nameko"):
        if forbidden in dependencies:
            violations.append(f"forbidden dependency present: {forbidden}")

    registry = load_experiment_registry(program_root)
    x10_live = registry["X-10"]["authorization_scopes"]["live_arbitrage"]
    if x10_live["authorized"] or not x10_live.get("permanent_no_go", False):
        violations.append("X-10 live arbitrage scope is not permanently closed")
    if registry["X-07"]["authorization_scopes"]["formal_result"]["authorized"]:
        violations.append("X-07 formal result is prematurely authorized")
    if registry["X-05"]["authorization_scopes"]["formal_result"]["authorized"]:
        violations.append("X-05 formal result is prematurely authorized")
    readme = (program_root / "README.md").read_text(encoding="utf-8")
    if "Performance or return claims in any README are never evidence" not in readme:
        violations.append("README evidence disclaimer is missing")
    return NoGoAuditReport(
        violations=tuple(violations),
        real_money_authorized=real_money_authorized,
        live_maker_present=live_maker_present,
        live_arbitrage_present=live_arbitrage_present,
        llm_hot_path_present=llm_hot_path_present,
    )


__all__ = [
    "ArtifactRegistryRow",
    "DatasetRegistryRow",
    "FormalResearchInputError",
    "ModelRegistryRow",
    "NoGoAuditReport",
    "ProgramAuditError",
    "ResearchRegistryError",
    "audit_no_go",
    "load_artifact_registry",
    "load_dataset_registry",
    "load_model_registry",
    "validate_formal_research_inputs",
    "validate_registered_research_bindings",
    "validate_research_inputs",
]
