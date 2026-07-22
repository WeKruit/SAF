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
from typing import Iterable, Mapping, Sequence

import prediction_market.contracts as contracts_module
import prediction_market.experiments as experiments_module
from prediction_market.contracts import canonical_sha256, validate_contract_v0
from prediction_market.experiments import load_experiment_registry


_PAIR_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
_MARKET_ID = re.compile(r"market_[A-Za-z0-9][A-Za-z0-9._:-]*")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_CONFIDENCE_TEXT = re.compile(r"(?:0(?:\.[0-9]+)?|1(?:\.0+)?)")
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


class MissingRecallDenominatorError(ClusterInputError):
    """Recall was requested without a registered candidate universe."""


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
        reason = "X-10 gate requires exactly 50 preregistered reviewed pairs"
    elif passed:
        reason = "precision gate passed for further G1 research only"
    else:
        reason = "precision below 90%; every cluster requires manual review"
    return ClusterGateDecision(
        may_advance=passed,
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


def compute_recall(
    reviewed_matches: set[str],
    *,
    candidate_universe: set[str] | None,
) -> Decimal:
    if candidate_universe is None:
        raise MissingRecallDenominatorError(
            "recall requires a preregistered candidate universe denominator"
        )
    if not candidate_universe:
        raise MissingRecallDenominatorError("candidate universe must not be empty")
    if not reviewed_matches <= candidate_universe:
        raise ClusterInputError("reviewed match is outside candidate universe")
    return Decimal(len(reviewed_matches)) / Decimal(len(candidate_universe))


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


def _x10_hashes(pairs: Sequence[ClusterPairV0]) -> tuple[str, str]:
    material = [
        {
            "pair_id": pair.pair_id,
            "left_market_id": pair.left_market_id,
            "right_market_id": pair.right_market_id,
            "relation_type": pair.relation_type,
            "confidence": str(pair.confidence),
            "router_version": pair.router_version,
            "observed_at": pair.observed_at,
        }
        for pair in sorted(pairs, key=lambda value: value.pair_id)
    ]
    files = {
        "clusters.py": Path(__file__),
        "contracts.py": Path(contracts_module.__file__),
        "experiments.py": Path(experiments_module.__file__),
    }
    code_manifest = {
        name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(files.items())
    }
    return canonical_sha256(code_manifest), canonical_sha256(material)


def run_x10_precision_audit(
    program_root: str | Path,
    *,
    pairs: Sequence[ClusterPairV0],
    adjudications: Mapping[str, bool],
    router_taxonomy_sha256: str | None = None,
    gold_protocol_sha256: str | None = None,
    h_approval_sha256: str | None = None,
) -> PrecisionReportV0:
    """Run only the precision scope after all X-10 evidence is preregistered."""

    registry = load_experiment_registry(program_root)
    card = registry["X-10"]
    scope = card["authorization_scopes"]["precision_audit"]
    if not scope["authorized"]:
        raise X10AuthorizationError("X-10 precision_audit is not authorized")
    locks = {lock["id"]: lock for lock in card["registration_locks"]}
    unresolved = [
        lock_id
        for lock_id in scope["required_lock_ids"]
        if locks[lock_id]["status"] != "resolved"
    ]
    if unresolved:
        raise X10AuthorizationError("X-10 unresolved locks: " + ", ".join(unresolved))
    if "precision_audit" not in card["preregistered_inputs"]:
        raise X10AuthorizationError("X-10 precision inputs are not preregistered")
    values = _validate_pair_set(pairs)
    frozen_adjudications = dict(adjudications)
    if len(values) != 50:
        raise X10AuthorizationError("X-10 requires exactly 50 preregistered pairs")
    code_hash, data_hash = _x10_hashes(values)
    preregistered = card["preregistered_inputs"]["precision_audit"]
    if preregistered["code_sha256"] != code_hash:
        raise X10AuthorizationError("X-10 runtime code differs from preregistration")
    if preregistered["data_sha256"] != data_hash:
        raise X10AuthorizationError("X-10 sample differs from preregistration")
    expected_evidence = {
        "matched_sample_registered": data_hash,
        "router_and_taxonomy_available": router_taxonomy_sha256,
        "gold_standard_protocol": gold_protocol_sha256,
        "h_split_approval": h_approval_sha256,
    }
    for lock_id, evidence in expected_evidence.items():
        if evidence is None or not _SHA256.fullmatch(evidence):
            raise X10AuthorizationError(f"X-10 {lock_id} evidence is required")
        if locks[lock_id].get("evidence_ref") != evidence:
            raise X10AuthorizationError(f"X-10 {lock_id} evidence mismatch")
    return compute_precision(values, frozen_adjudications)


__all__ = [
    "ClusterGateDecision",
    "ClusterInputError",
    "ClusterPairV0",
    "MissingRecallDenominatorError",
    "PrecisionReportV0",
    "ReviewQueueItemV0",
    "X10AuthorizationError",
    "build_review_queue",
    "cluster_gate",
    "compute_precision",
    "compute_recall",
    "load_cluster_pairs_csv",
    "run_x10_precision_audit",
]
