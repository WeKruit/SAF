"""Program-level artifact and NO-GO audit for the first-round handoff."""

from __future__ import annotations

import ast
import csv
import io
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from prediction_market.compliance import load_compliance_matrix, may_execute_real_money
from prediction_market.experiments import load_experiment_registry


_ARTIFACT_COLUMNS = (
    "artifact_id",
    "path",
    "owner_team",
    "version",
    "due_gate",
    "status",
)
_ARTIFACT_ID = re.compile(r"ART-[A-Z0-9][A-Z0-9-]*")
_VERSION = re.compile(r"v[0-9]+(?:\.[0-9]+)?")
_ALLOWED_STATUS = frozenset(
    {"registered", "complete", "blocked", "in_progress", "harness_pass"}
)


class ProgramAuditError(ValueError):
    """A program artifact registry cannot be trusted."""


@dataclass(frozen=True, slots=True)
class ArtifactRegistryRow:
    artifact_id: str
    path: str
    owner_team: str
    version: str
    due_gate: str
    status: str


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


def audit_no_go(root: str | Path) -> NoGoAuditReport:
    """Verify executable surfaces remain inside Charter v0.2 blocked scope."""

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
    "NoGoAuditReport",
    "ProgramAuditError",
    "audit_no_go",
    "load_artifact_registry",
]
