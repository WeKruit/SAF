"""Matched-cluster review workflow for the registered X-10 audit."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

import prediction_market.contracts as contracts_module
import prediction_market.experiments as experiments_module
from prediction_market.contracts import (
    canonical_json_bytes,
    canonical_sha256,
    validate_contract_v0,
)
from prediction_market.experiments import load_experiment_registry


_PAIR_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
_MARKET_ID = re.compile(r"market_[A-Za-z0-9][A-Za-z0-9._:-]*")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_CONFIDENCE_TEXT = re.compile(r"(?:0(?:\.[0-9]+)?|1(?:\.0+)?)")
_X10_SELECTION_METHOD = "sha256_seeded_rank_v0"
_MARKET_RELATIONS_CONTRACT = Path("contracts/market-relations/v0.yaml")
_GOLD_PROTOCOL_CONTRACT = Path("contracts/x10-gold-adjudication/v0.yaml")
_CSV_COLUMNS = (
    "pair_id",
    "left_market_id",
    "right_market_id",
    "relation_type",
    "confidence",
    "router_version",
    "observed_at",
)


class ClusterInputError(ValueError):
    """Cluster evidence is malformed or cannot support the requested metric."""


class X10AuthorizationError(PermissionError):
    """X-10 locks or preregistration do not authorize the formal audit."""


@dataclass(frozen=True, slots=True)
class ClusterPairV0:
    pair_id: str
    left_market_id: str
    right_market_id: str
    relation_type: str
    confidence: Decimal
    router_version: str
    observed_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not _PAIR_ID.fullmatch(self.pair_id):
            raise ClusterInputError("pair_id must be canonical")
        for name, value in (
            ("left_market_id", self.left_market_id),
            ("right_market_id", self.right_market_id),
        ):
            if not isinstance(value, str) or not _MARKET_ID.fullmatch(value):
                raise ClusterInputError(f"{name} must be a canonical market ID")
        if self.left_market_id == self.right_market_id:
            raise ClusterInputError("cluster pair must contain two distinct markets")
        try:
            validate_contract_v0("market-relations/v0.yaml", self.relation_type)
        except (TypeError, ValueError) as error:
            raise ClusterInputError("relation_type is outside the v0 taxonomy") from error
        if not isinstance(self.confidence, Decimal) or not self.confidence.is_finite():
            raise ClusterInputError("confidence must be a finite Decimal")
        if not Decimal(0) <= self.confidence <= Decimal(1):
            raise ClusterInputError("confidence must be in [0, 1]")
        if (
            not isinstance(self.router_version, str)
            or not self.router_version
            or self.router_version != self.router_version.strip()
        ):
            raise ClusterInputError("router_version must be explicit")
        if not isinstance(self.observed_at, str) or not _UTC.fullmatch(self.observed_at):
            raise ClusterInputError("observed_at must be canonical UTC")
        try:
            datetime.fromisoformat(self.observed_at.removesuffix("Z") + "+00:00")
        except ValueError as error:
            raise ClusterInputError("observed_at is not a valid UTC instant") from error


@dataclass(frozen=True, slots=True)
class ReviewQueueItemV0:
    pair_id: str
    priority: int
    status: str
    confidence: Decimal
    relation_type: str


@dataclass(frozen=True, slots=True)
class ClusterGateDecision:
    numeric_threshold_met: bool
    may_advance: bool
    live_arbitrage_authorized: bool
    correct: int
    reviewed: int
    reason: str


@dataclass(frozen=True, slots=True)
class PrecisionReportV0:
    correct: int
    reviewed: int
    precision: Decimal
    gate: ClusterGateDecision


@dataclass(frozen=True, slots=True)
class ConfidenceBinV0:
    lower: Decimal
    upper: Decimal
    upper_inclusive: bool
    count: int
    mean_confidence: Decimal | None
    observed_precision: Decimal | None


@dataclass(frozen=True, slots=True)
class ConfidenceCalibrationV0:
    bins: tuple[ConfidenceBinV0, ...]
    expected_calibration_error: Decimal


@dataclass(frozen=True, slots=True)
class SemanticAdjudicationV0:
    is_correct: bool
    semantic_difference_category: str

    def __post_init__(self) -> None:
        if type(self.is_correct) is not bool:
            raise ClusterInputError("is_correct must be a boolean")
        if (
            not isinstance(self.semantic_difference_category, str)
            or not self.semantic_difference_category
            or self.semantic_difference_category
            != self.semantic_difference_category.strip()
        ):
            raise ClusterInputError("semantic_difference_category must be explicit")


@dataclass(frozen=True, slots=True)
class SemanticDifferenceCountV0:
    category: str
    count: int


@dataclass(frozen=True, slots=True)
class X10PrecisionPreregistrationV0:
    selected_pairs: tuple[ClusterPairV0, ...]
    selection_method: str
    selection_seed: int
    router_query: str
    router_version: str
    confidence_bin_edges: tuple[Decimal, ...]
    semantic_difference_categories: tuple[str, ...]
    code_manifest_bytes: bytes
    sample_manifest_bytes: bytes
    router_taxonomy_evidence_bytes: bytes
    code_sha256: str
    data_sha256: str
    router_taxonomy_evidence_sha256: str
    gold_protocol_sha256: str


@dataclass(frozen=True, slots=True)
class X10RecallPreregistrationV0:
    candidate_universe: tuple[ClusterPairV0, ...]
    router_query: str
    router_version: str
    code_manifest_bytes: bytes
    denominator_manifest_bytes: bytes
    router_taxonomy_evidence_bytes: bytes
    code_sha256: str
    data_sha256: str
    router_taxonomy_evidence_sha256: str
    gold_protocol_sha256: str


@dataclass(frozen=True, slots=True)
class X10PrecisionAuditReportV0:
    preregistration: X10PrecisionPreregistrationV0
    precision: PrecisionReportV0
    calibration: ConfidenceCalibrationV0
    semantic_differences: tuple[SemanticDifferenceCountV0, ...]
    review_queue: tuple[ReviewQueueItemV0, ...]
    g1_research_decision: str
    live_arbitrage_authorized: bool
    h_approval_sha256: str
    evaluation_started_at: str
    registration_head_sha256: str
    result_sha256: str


@dataclass(frozen=True, slots=True)
class X10RecallAuditReportV0:
    preregistration: X10RecallPreregistrationV0
    recalled: int
    denominator: int
    recall: Decimal
    live_arbitrage_authorized: bool
    h_approval_sha256: str
    evaluation_started_at: str
    registration_head_sha256: str
    result_sha256: str


def _validate_pair_set(pairs: Iterable[ClusterPairV0]) -> tuple[ClusterPairV0, ...]:
    values = tuple(pairs)
    if any(not isinstance(pair, ClusterPairV0) for pair in values):
        raise ClusterInputError("all pairs must be ClusterPairV0")
    ids = [pair.pair_id for pair in values]
    if len(ids) != len(set(ids)):
        raise ClusterInputError("pair_id values must be unique")
    native_pairs = [
        tuple(sorted((pair.left_market_id, pair.right_market_id))) for pair in values
    ]
    if len(native_pairs) != len(set(native_pairs)):
        raise ClusterInputError("market pairs must be unique")
    return values


def build_review_queue(
    pairs: Iterable[ClusterPairV0],
) -> tuple[ReviewQueueItemV0, ...]:
    """Prioritize low-confidence assertions for deterministic manual review."""

    ordered = sorted(
        _validate_pair_set(pairs), key=lambda pair: (pair.confidence, pair.pair_id)
    )
    return tuple(
        ReviewQueueItemV0(
            pair_id=pair.pair_id,
            priority=index,
            status="PENDING_MANUAL_REVIEW",
            confidence=pair.confidence,
            relation_type=pair.relation_type,
        )
        for index, pair in enumerate(ordered, start=1)
    )


def cluster_gate(*, correct: int, reviewed: int) -> ClusterGateDecision:
    if type(correct) is not int or type(reviewed) is not int:
        raise ClusterInputError("correct and reviewed must be integers")
    if correct < 0 or reviewed < 0 or correct > reviewed:
        raise ClusterInputError("invalid adjudication counts")
    exact_sample = reviewed == 50
    passed = exact_sample and correct >= 45
    if not exact_sample:
        reason = "descriptive threshold requires exactly 50 reviewed pairs"
    elif passed:
        reason = "descriptive numeric threshold met; no authorization was evaluated"
    else:
        reason = "descriptive numeric threshold not met"
    return ClusterGateDecision(
        numeric_threshold_met=passed,
        may_advance=False,
        live_arbitrage_authorized=False,
        correct=correct,
        reviewed=reviewed,
        reason=reason,
    )


def compute_precision(
    pairs: Iterable[ClusterPairV0],
    adjudications: Mapping[str, bool],
) -> PrecisionReportV0:
    values = _validate_pair_set(pairs)
    expected = {pair.pair_id for pair in values}
    if set(adjudications) != expected or len(adjudications) != len(values):
        raise ClusterInputError("every pair must be adjudicated exactly once")
    if any(type(value) is not bool for value in adjudications.values()):
        raise ClusterInputError("adjudications must be booleans")
    reviewed = len(values)
    if reviewed == 0:
        raise ClusterInputError("precision requires at least one pair")
    correct = sum(adjudications.values())
    return PrecisionReportV0(
        correct=correct,
        reviewed=reviewed,
        precision=Decimal(correct) / Decimal(reviewed),
        gate=cluster_gate(correct=correct, reviewed=reviewed),
    )


def confidence_calibration(
    pairs: Iterable[ClusterPairV0],
    adjudications: Mapping[str, bool],
    *,
    bin_edges: tuple[Decimal, ...],
) -> ConfidenceCalibrationV0:
    """Summarize correctness in preregistered confidence bins."""

    values = _validate_pair_set(pairs)
    frozen = dict(adjudications)
    expected = {pair.pair_id for pair in values}
    if set(frozen) != expected or len(frozen) != len(values):
        raise ClusterInputError("every pair must be adjudicated exactly once")
    if any(type(value) is not bool for value in frozen.values()):
        raise ClusterInputError("adjudications must be booleans")
    if (
        type(bin_edges) is not tuple
        or len(bin_edges) < 2
        or any(not isinstance(edge, Decimal) or not edge.is_finite() for edge in bin_edges)
        or bin_edges[0] != Decimal(0)
        or bin_edges[-1] != Decimal(1)
        or any(left >= right for left, right in zip(bin_edges, bin_edges[1:]))
    ):
        raise ClusterInputError(
            "bin_edges must be an explicit strictly increasing Decimal tuple from 0 to 1"
        )
    bins: list[ConfidenceBinV0] = []
    total = Decimal(len(values))
    ece = Decimal(0)
    for index, (lower, upper) in enumerate(
        zip(bin_edges, bin_edges[1:])
    ):
        inclusive = index == len(bin_edges) - 2
        members = [
            pair
            for pair in values
            if pair.confidence >= lower
            and (pair.confidence <= upper if inclusive else pair.confidence < upper)
        ]
        if members:
            count = len(members)
            mean_confidence = sum(
                (pair.confidence for pair in members), start=Decimal(0)
            ) / Decimal(count)
            observed = Decimal(
                sum(1 for pair in members if frozen[pair.pair_id])
            ) / Decimal(count)
            ece += Decimal(count) / total * abs(mean_confidence - observed)
        else:
            count = 0
            mean_confidence = None
            observed = None
        bins.append(
            ConfidenceBinV0(
                lower=lower,
                upper=upper,
                upper_inclusive=inclusive,
                count=count,
                mean_confidence=mean_confidence,
                observed_precision=observed,
            )
        )
    return ConfidenceCalibrationV0(
        bins=tuple(bins), expected_calibration_error=ece
    )


def load_cluster_pairs_csv(path: str | Path) -> tuple[ClusterPairV0, ...]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ClusterInputError("cluster CSV must be a regular non-symlink file")
    try:
        text = source.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ClusterInputError("cluster CSV must be readable UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if tuple(reader.fieldnames or ()) != _CSV_COLUMNS:
        raise ClusterInputError("cluster CSV columns do not match v0")
    pairs: list[ClusterPairV0] = []
    try:
        for row in reader:
            if None in row or any(
                row[field] is None or row[field] != row[field].strip()
                for field in _CSV_COLUMNS
            ):
                raise ClusterInputError("cluster CSV has non-canonical cells")
            try:
                if not _CONFIDENCE_TEXT.fullmatch(row["confidence"]):
                    raise ClusterInputError("confidence is not a canonical Decimal")
                confidence = Decimal(row["confidence"])
            except Exception as error:
                raise ClusterInputError("confidence is not a canonical Decimal") from error
            pairs.append(
                ClusterPairV0(
                    pair_id=row["pair_id"],
                    left_market_id=row["left_market_id"],
                    right_market_id=row["right_market_id"],
                    relation_type=row["relation_type"],
                    confidence=confidence,
                    router_version=row["router_version"],
                    observed_at=row["observed_at"],
                )
            )
    except csv.Error as error:
        raise ClusterInputError(f"invalid cluster CSV: {error}") from error
    return _validate_pair_set(pairs)


def _raw_sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _read_regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise X10AuthorizationError(f"X-10 {label} must be a regular non-symlink file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise X10AuthorizationError(f"X-10 {label} is unreadable") from error


def _pair_material(pair: ClusterPairV0) -> dict[str, str]:
    return {
        "pair_id": pair.pair_id,
        "left_market_id": pair.left_market_id,
        "right_market_id": pair.right_market_id,
        "relation_type": pair.relation_type,
        "confidence": str(pair.confidence),
        "router_version": pair.router_version,
        "observed_at": pair.observed_at,
    }


@dataclass(frozen=True, slots=True)
class _BoundX10Evidence:
    code_manifest_bytes: bytes
    router_taxonomy_evidence_bytes: bytes
    code_sha256: str
    router_taxonomy_evidence_sha256: str
    gold_protocol_sha256: str
    confidence_bin_edges: tuple[Decimal, ...]
    semantic_difference_categories: tuple[str, ...]


def _load_gold_protocol(
    program_root: Path,
) -> tuple[bytes, tuple[Decimal, ...], tuple[str, ...]]:
    raw = _read_regular_bytes(
        program_root / _GOLD_PROTOCOL_CONTRACT, "gold adjudication protocol"
    )
    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise X10AuthorizationError("X-10 gold adjudication protocol is invalid YAML") from error
    expected_keys = {
        "title",
        "contract_version",
        "required_adjudication_fields",
        "confidence_bin_edges",
        "semantic_difference_categories",
    }
    if type(document) is not dict or set(document) != expected_keys:
        raise X10AuthorizationError("X-10 gold adjudication protocol has invalid fields")
    if document["contract_version"] != "v0" or document[
        "required_adjudication_fields"
    ] != ["pair_id", "is_correct", "semantic_difference_category"]:
        raise X10AuthorizationError("X-10 gold adjudication protocol is not v0")
    raw_edges = document["confidence_bin_edges"]
    if (
        type(raw_edges) is not list
        or len(raw_edges) < 2
        or any(type(edge) is not str or not _CONFIDENCE_TEXT.fullmatch(edge) for edge in raw_edges)
    ):
        raise X10AuthorizationError("X-10 confidence bin edges are invalid")
    edges = tuple(Decimal(edge) for edge in raw_edges)
    if (
        edges[0] != Decimal(0)
        or edges[-1] != Decimal(1)
        or any(left >= right for left, right in zip(edges, edges[1:]))
    ):
        raise X10AuthorizationError("X-10 confidence bin edges are invalid")
    raw_categories = document["semantic_difference_categories"]
    if (
        type(raw_categories) is not list
        or not raw_categories
        or any(
            type(category) is not str
            or not category
            or category != category.strip()
            for category in raw_categories
        )
        or len(raw_categories) != len(set(raw_categories))
        or "none" not in raw_categories
    ):
        raise X10AuthorizationError("X-10 semantic difference categories are invalid")
    return raw, edges, tuple(raw_categories)


def _explicit_router_value(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ClusterInputError(f"{label} must be explicit")
    return value


def _bound_x10_evidence(
    program_root: Path, *, router_query: str, router_version: str
) -> _BoundX10Evidence:
    router_query = _explicit_router_value(router_query, "router_query")
    router_version = _explicit_router_value(router_version, "router_version")
    taxonomy_path = program_root / _MARKET_RELATIONS_CONTRACT
    taxonomy_raw = _read_regular_bytes(taxonomy_path, "market relation taxonomy")
    gold_raw, bin_edges, categories = _load_gold_protocol(program_root)
    module_paths = {
        "src/prediction_market/clusters.py": Path(__file__),
        "src/prediction_market/contracts.py": Path(contracts_module.__file__),
        "src/prediction_market/experiments.py": Path(experiments_module.__file__),
    }
    code_manifest: dict[str, str] = {
        name: _raw_sha256(_read_regular_bytes(path, name))
        for name, path in sorted(module_paths.items())
    }
    taxonomy_sha256 = _raw_sha256(taxonomy_raw)
    gold_sha256 = _raw_sha256(gold_raw)
    code_manifest[_MARKET_RELATIONS_CONTRACT.as_posix()] = taxonomy_sha256
    code_manifest[_GOLD_PROTOCOL_CONTRACT.as_posix()] = gold_sha256
    code_manifest_bytes = canonical_json_bytes(code_manifest)
    router_evidence_bytes = canonical_json_bytes(
        {
            "router_query": router_query,
            "router_version": router_version,
            "taxonomy_sha256": taxonomy_sha256,
        }
    )
    return _BoundX10Evidence(
        code_manifest_bytes=code_manifest_bytes,
        router_taxonomy_evidence_bytes=router_evidence_bytes,
        code_sha256=_raw_sha256(code_manifest_bytes),
        router_taxonomy_evidence_sha256=_raw_sha256(router_evidence_bytes),
        gold_protocol_sha256=gold_sha256,
        confidence_bin_edges=bin_edges,
        semantic_difference_categories=categories,
    )


def _validate_candidate_router_version(
    pairs: Sequence[ClusterPairV0], router_version: str
) -> None:
    if any(pair.router_version != router_version for pair in pairs):
        raise ClusterInputError("every candidate pair must match the registered router_version")


def build_x10_precision_preregistration(
    program_root: str | Path,
    *,
    candidate_universe: Sequence[ClusterPairV0],
    selection_method: str,
    selection_seed: int,
    router_query: str,
    router_version: str,
) -> X10PrecisionPreregistrationV0:
    """Build the exact immutable bytes that must be registered before X-10 review."""

    values = _validate_pair_set(candidate_universe)
    if len(values) < 50:
        raise ClusterInputError("X-10 candidate universe must contain at least 50 pairs")
    if selection_method != _X10_SELECTION_METHOD:
        raise ClusterInputError(f"selection_method must be {_X10_SELECTION_METHOD}")
    if type(selection_seed) is not int or not 0 <= selection_seed <= 2**63 - 1:
        raise ClusterInputError("selection_seed must be a non-negative 64-bit integer")
    router_query = _explicit_router_value(router_query, "router_query")
    router_version = _explicit_router_value(router_version, "router_version")
    _validate_candidate_router_version(values, router_version)
    ranked = sorted(
        values,
        key=lambda pair: (
            canonical_sha256(
                {
                    "domain": _X10_SELECTION_METHOD,
                    "seed": selection_seed,
                    "pair": _pair_material(pair),
                }
            ),
            pair.pair_id,
        ),
    )
    selected = tuple(ranked[:50])
    candidate_material = [
        _pair_material(pair) for pair in sorted(values, key=lambda pair: pair.pair_id)
    ]
    sample_manifest_bytes = canonical_json_bytes(
        {
            "schema": "x10_precision_preregistration_v0",
            "candidate_universe": candidate_material,
            "router_query": router_query,
            "router_version": router_version,
            "selection": {
                "method": selection_method,
                "seed": selection_seed,
                "selected_pair_ids": [pair.pair_id for pair in selected],
            },
            "selected_pairs": [_pair_material(pair) for pair in selected],
        }
    )
    evidence = _bound_x10_evidence(
        Path(program_root), router_query=router_query, router_version=router_version
    )
    return X10PrecisionPreregistrationV0(
        selected_pairs=selected,
        selection_method=selection_method,
        selection_seed=selection_seed,
        router_query=router_query,
        router_version=router_version,
        confidence_bin_edges=evidence.confidence_bin_edges,
        semantic_difference_categories=evidence.semantic_difference_categories,
        code_manifest_bytes=evidence.code_manifest_bytes,
        sample_manifest_bytes=sample_manifest_bytes,
        router_taxonomy_evidence_bytes=evidence.router_taxonomy_evidence_bytes,
        code_sha256=evidence.code_sha256,
        data_sha256=_raw_sha256(sample_manifest_bytes),
        router_taxonomy_evidence_sha256=evidence.router_taxonomy_evidence_sha256,
        gold_protocol_sha256=evidence.gold_protocol_sha256,
    )


def build_x10_recall_preregistration(
    program_root: str | Path,
    *,
    candidate_universe: Sequence[ClusterPairV0],
    router_query: str,
    router_version: str,
) -> X10RecallPreregistrationV0:
    """Build the exact candidate-universe denominator bytes for formal recall."""

    values = _validate_pair_set(candidate_universe)
    if not values:
        raise ClusterInputError("recall candidate universe must not be empty")
    router_query = _explicit_router_value(router_query, "router_query")
    router_version = _explicit_router_value(router_version, "router_version")
    _validate_candidate_router_version(values, router_version)
    denominator_manifest_bytes = canonical_json_bytes(
        {
            "schema": "x10_recall_denominator_v0",
            "candidate_universe": [
                _pair_material(pair)
                for pair in sorted(values, key=lambda pair: pair.pair_id)
            ],
            "router_query": router_query,
            "router_version": router_version,
        }
    )
    evidence = _bound_x10_evidence(
        Path(program_root), router_query=router_query, router_version=router_version
    )
    return X10RecallPreregistrationV0(
        candidate_universe=tuple(values),
        router_query=router_query,
        router_version=router_version,
        code_manifest_bytes=evidence.code_manifest_bytes,
        denominator_manifest_bytes=denominator_manifest_bytes,
        router_taxonomy_evidence_bytes=evidence.router_taxonomy_evidence_bytes,
        code_sha256=evidence.code_sha256,
        data_sha256=_raw_sha256(denominator_manifest_bytes),
        router_taxonomy_evidence_sha256=evidence.router_taxonomy_evidence_sha256,
        gold_protocol_sha256=evidence.gold_protocol_sha256,
    )


def _authorized_scope(
    card: Mapping[str, Any], scope_name: str
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]], Mapping[str, str]]:
    try:
        scope = card["authorization_scopes"][scope_name]
    except (KeyError, TypeError) as error:
        raise X10AuthorizationError(f"X-10 {scope_name} scope is absent") from error
    if scope.get("authorized") is not True or scope.get("permanent_no_go", False) is True:
        raise X10AuthorizationError(f"X-10 {scope_name} is not authorized")
    live_scope = card.get("authorization_scopes", {}).get("live_arbitrage")
    if (
        type(live_scope) is not dict
        or live_scope.get("authorized") is not False
        or live_scope.get("permanent_no_go") is not True
    ):
        raise X10AuthorizationError("X-10 live_arbitrage must remain permanent NO-GO")
    locks = {
        lock["id"]: lock
        for lock in card.get("registration_locks", [])
        if type(lock) is dict and "id" in lock
    }
    try:
        required_lock_ids = tuple(scope["required_lock_ids"])
    except (KeyError, TypeError) as error:
        raise X10AuthorizationError(f"X-10 {scope_name} locks are malformed") from error
    unresolved = [
        lock_id
        for lock_id in required_lock_ids
        if lock_id not in locks or locks[lock_id].get("status") != "resolved"
    ]
    if unresolved:
        raise X10AuthorizationError("X-10 unresolved locks: " + ", ".join(unresolved))
    try:
        preregistered = card["preregistered_inputs"][scope_name]
    except (KeyError, TypeError) as error:
        raise X10AuthorizationError(f"X-10 {scope_name} inputs are not preregistered") from error
    return scope, locks, preregistered


def _verify_registration_head(
    card: Mapping[str, Any], registration_head_sha256: str
) -> None:
    if (
        not isinstance(registration_head_sha256, str)
        or not _SHA256.fullmatch(registration_head_sha256)
        or card.get("registration_head_sha256") != registration_head_sha256
    ):
        raise X10AuthorizationError("X-10 registration head mismatch")


def _verify_preregistered_hashes(
    preregistered: Mapping[str, str], *, code_sha256: str, data_sha256: str
) -> None:
    if preregistered.get("code_sha256") != code_sha256:
        raise X10AuthorizationError("X-10 runtime code/protocol differs from preregistration")
    if preregistered.get("data_sha256") != data_sha256:
        raise X10AuthorizationError("X-10 registered data material differs from runtime bytes")


def _verify_lock_evidence(
    locks: Mapping[str, Mapping[str, Any]], expected: Mapping[str, str]
) -> None:
    for lock_id, evidence_sha256 in expected.items():
        lock = locks.get(lock_id)
        if lock is None or lock.get("evidence_ref") != evidence_sha256:
            raise X10AuthorizationError(f"X-10 {lock_id} evidence mismatch")


def _validate_result_reference(
    program_root: str | Path,
    *,
    scope_name: str,
    result_label: str,
    evaluation_started_at: str,
    code_sha256: str,
    data_sha256: str,
    result_sha256: str,
    registration_head_sha256: str,
) -> None:
    result_ref = {
        "scope": scope_name,
        "result_label": result_label,
        "evaluation_started_at": evaluation_started_at,
        "code_sha256": code_sha256,
        "data_sha256": data_sha256,
        "result_sha256": result_sha256,
        "registration_head_sha256": registration_head_sha256,
    }
    try:
        experiments_module.validate_result_ref(program_root, "X-10", result_ref)
    except experiments_module.ExperimentRegistryError as error:
        raise X10AuthorizationError(f"X-10 formal result rejected: {error}") from error


def _calibration_payload(calibration: ConfidenceCalibrationV0) -> dict[str, Any]:
    return {
        "bins": [
            {
                "lower": str(item.lower),
                "upper": str(item.upper),
                "upper_inclusive": item.upper_inclusive,
                "count": item.count,
                "mean_confidence": (
                    None if item.mean_confidence is None else str(item.mean_confidence)
                ),
                "observed_precision": (
                    None
                    if item.observed_precision is None
                    else str(item.observed_precision)
                ),
            }
            for item in calibration.bins
        ],
        "expected_calibration_error": str(calibration.expected_calibration_error),
    }


def run_x10_precision_audit(
    program_root: str | Path,
    *,
    candidate_universe: Sequence[ClusterPairV0],
    selection_method: str,
    selection_seed: int,
    router_query: str,
    router_version: str,
    adjudications: Mapping[str, SemanticAdjudicationV0],
    h_approval_path: str | Path,
    evaluation_started_at: str,
    registration_head_sha256: str,
) -> X10PrecisionAuditReportV0:
    """Return the only X-10 object allowed to make a G1 research decision."""

    registry = load_experiment_registry(program_root)
    card = registry["X-10"]
    scope, locks, registered = _authorized_scope(card, "precision_audit")
    _verify_registration_head(card, registration_head_sha256)
    preregistration = build_x10_precision_preregistration(
        program_root,
        candidate_universe=candidate_universe,
        selection_method=selection_method,
        selection_seed=selection_seed,
        router_query=router_query,
        router_version=router_version,
    )
    _verify_preregistered_hashes(
        registered,
        code_sha256=preregistration.code_sha256,
        data_sha256=preregistration.data_sha256,
    )
    h_approval_sha256 = _raw_sha256(
        _read_regular_bytes(Path(h_approval_path), "H split approval")
    )
    _verify_lock_evidence(
        locks,
        {
            "matched_sample_registered": preregistration.data_sha256,
            "router_and_taxonomy_available": (
                preregistration.router_taxonomy_evidence_sha256
            ),
            "gold_standard_protocol": preregistration.gold_protocol_sha256,
            "h_split_approval": h_approval_sha256,
        },
    )
    result_label = scope["required_result_label"]
    _validate_result_reference(
        program_root,
        scope_name="precision_audit",
        result_label=result_label,
        evaluation_started_at=evaluation_started_at,
        code_sha256=preregistration.code_sha256,
        data_sha256=preregistration.data_sha256,
        result_sha256="sha256:" + "0" * 64,
        registration_head_sha256=registration_head_sha256,
    )

    frozen_adjudications = dict(adjudications)
    expected_ids = {pair.pair_id for pair in preregistration.selected_pairs}
    if set(frozen_adjudications) != expected_ids or len(frozen_adjudications) != 50:
        raise ClusterInputError("every selected X-10 pair must be adjudicated exactly once")
    if any(
        not isinstance(value, SemanticAdjudicationV0)
        for value in frozen_adjudications.values()
    ):
        raise ClusterInputError("formal adjudications must be SemanticAdjudicationV0")
    allowed_categories = set(preregistration.semantic_difference_categories)
    if any(
        value.semantic_difference_category not in allowed_categories
        for value in frozen_adjudications.values()
    ):
        raise ClusterInputError("semantic difference category is outside the gold protocol")
    correctness = {
        pair_id: adjudication.is_correct
        for pair_id, adjudication in frozen_adjudications.items()
    }
    precision = compute_precision(preregistration.selected_pairs, correctness)
    calibration = confidence_calibration(
        preregistration.selected_pairs,
        correctness,
        bin_edges=preregistration.confidence_bin_edges,
    )
    semantic_differences = tuple(
        SemanticDifferenceCountV0(
            category=category,
            count=sum(
                1
                for adjudication in frozen_adjudications.values()
                if adjudication.semantic_difference_category == category
            ),
        )
        for category in preregistration.semantic_difference_categories
    )
    review_queue = build_review_queue(preregistration.selected_pairs)
    decision = (
        "G1_RESEARCH_ADVANCE"
        if precision.gate.numeric_threshold_met
        else "G1_RESEARCH_NO_GO"
    )
    result_payload = {
        "schema": "x10_precision_formal_result_v0",
        "code_sha256": preregistration.code_sha256,
        "data_sha256": preregistration.data_sha256,
        "router_taxonomy_evidence_sha256": (
            preregistration.router_taxonomy_evidence_sha256
        ),
        "gold_protocol_sha256": preregistration.gold_protocol_sha256,
        "h_approval_sha256": h_approval_sha256,
        "evaluation_started_at": evaluation_started_at,
        "registration_head_sha256": registration_head_sha256,
        "precision": {
            "correct": precision.correct,
            "reviewed": precision.reviewed,
            "precision": str(precision.precision),
            "numeric_threshold_met": precision.gate.numeric_threshold_met,
            "may_advance": precision.gate.may_advance,
        },
        "calibration": _calibration_payload(calibration),
        "semantic_differences": [
            {"category": item.category, "count": item.count}
            for item in semantic_differences
        ],
        "review_queue": [
            {
                "pair_id": item.pair_id,
                "priority": item.priority,
                "status": item.status,
                "confidence": str(item.confidence),
                "relation_type": item.relation_type,
            }
            for item in review_queue
        ],
        "g1_research_decision": decision,
        "live_arbitrage_authorized": False,
    }
    result_sha256 = canonical_sha256(result_payload)
    _validate_result_reference(
        program_root,
        scope_name="precision_audit",
        result_label=result_label,
        evaluation_started_at=evaluation_started_at,
        code_sha256=preregistration.code_sha256,
        data_sha256=preregistration.data_sha256,
        result_sha256=result_sha256,
        registration_head_sha256=registration_head_sha256,
    )
    return X10PrecisionAuditReportV0(
        preregistration=preregistration,
        precision=precision,
        calibration=calibration,
        semantic_differences=semantic_differences,
        review_queue=review_queue,
        g1_research_decision=decision,
        live_arbitrage_authorized=False,
        h_approval_sha256=h_approval_sha256,
        evaluation_started_at=evaluation_started_at,
        registration_head_sha256=registration_head_sha256,
        result_sha256=result_sha256,
    )


def run_x10_recall_audit(
    program_root: str | Path,
    *,
    candidate_universe: Sequence[ClusterPairV0],
    reviewed_match_ids: set[str],
    router_query: str,
    router_version: str,
    h_approval_path: str | Path,
    evaluation_started_at: str,
    registration_head_sha256: str,
) -> X10RecallAuditReportV0:
    """Compute recall only through the authorized, preregistered X-10 scope."""

    registry = load_experiment_registry(program_root)
    card = registry["X-10"]
    scope, locks, registered = _authorized_scope(card, "recall")
    _verify_registration_head(card, registration_head_sha256)
    preregistration = build_x10_recall_preregistration(
        program_root,
        candidate_universe=candidate_universe,
        router_query=router_query,
        router_version=router_version,
    )
    _verify_preregistered_hashes(
        registered,
        code_sha256=preregistration.code_sha256,
        data_sha256=preregistration.data_sha256,
    )
    h_approval_sha256 = _raw_sha256(
        _read_regular_bytes(Path(h_approval_path), "H split approval")
    )
    _verify_lock_evidence(
        locks,
        {
            "recall_candidate_universe": preregistration.data_sha256,
            "router_and_taxonomy_available": (
                preregistration.router_taxonomy_evidence_sha256
            ),
            "gold_standard_protocol": preregistration.gold_protocol_sha256,
            "h_split_approval": h_approval_sha256,
        },
    )
    result_label = scope["required_result_label"]
    _validate_result_reference(
        program_root,
        scope_name="recall",
        result_label=result_label,
        evaluation_started_at=evaluation_started_at,
        code_sha256=preregistration.code_sha256,
        data_sha256=preregistration.data_sha256,
        result_sha256="sha256:" + "0" * 64,
        registration_head_sha256=registration_head_sha256,
    )

    if type(reviewed_match_ids) is not set or any(
        type(pair_id) is not str for pair_id in reviewed_match_ids
    ):
        raise ClusterInputError("reviewed_match_ids must be a set of pair IDs")
    frozen_match_ids = set(reviewed_match_ids)
    candidate_ids = {pair.pair_id for pair in preregistration.candidate_universe}
    if not frozen_match_ids <= candidate_ids:
        raise ClusterInputError("reviewed match is outside the preregistered candidate universe")
    recalled = len(frozen_match_ids)
    denominator = len(candidate_ids)
    recall = Decimal(recalled) / Decimal(denominator)
    result_payload = {
        "schema": "x10_recall_formal_result_v0",
        "code_sha256": preregistration.code_sha256,
        "data_sha256": preregistration.data_sha256,
        "router_taxonomy_evidence_sha256": (
            preregistration.router_taxonomy_evidence_sha256
        ),
        "gold_protocol_sha256": preregistration.gold_protocol_sha256,
        "h_approval_sha256": h_approval_sha256,
        "evaluation_started_at": evaluation_started_at,
        "registration_head_sha256": registration_head_sha256,
        "reviewed_match_ids": sorted(frozen_match_ids),
        "recalled": recalled,
        "denominator": denominator,
        "recall": str(recall),
        "live_arbitrage_authorized": False,
    }
    result_sha256 = canonical_sha256(result_payload)
    _validate_result_reference(
        program_root,
        scope_name="recall",
        result_label=result_label,
        evaluation_started_at=evaluation_started_at,
        code_sha256=preregistration.code_sha256,
        data_sha256=preregistration.data_sha256,
        result_sha256=result_sha256,
        registration_head_sha256=registration_head_sha256,
    )
    return X10RecallAuditReportV0(
        preregistration=preregistration,
        recalled=recalled,
        denominator=denominator,
        recall=recall,
        live_arbitrage_authorized=False,
        h_approval_sha256=h_approval_sha256,
        evaluation_started_at=evaluation_started_at,
        registration_head_sha256=registration_head_sha256,
        result_sha256=result_sha256,
    )


__all__ = [
    "ClusterGateDecision",
    "ClusterInputError",
    "ClusterPairV0",
    "ConfidenceBinV0",
    "ConfidenceCalibrationV0",
    "PrecisionReportV0",
    "ReviewQueueItemV0",
    "SemanticAdjudicationV0",
    "SemanticDifferenceCountV0",
    "X10AuthorizationError",
    "X10PrecisionAuditReportV0",
    "X10PrecisionPreregistrationV0",
    "X10RecallAuditReportV0",
    "X10RecallPreregistrationV0",
    "build_review_queue",
    "build_x10_precision_preregistration",
    "build_x10_recall_preregistration",
    "cluster_gate",
    "confidence_calibration",
    "compute_precision",
    "load_cluster_pairs_csv",
    "run_x10_precision_audit",
    "run_x10_recall_audit",
]
