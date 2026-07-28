"""Fail-closed loader for the NFL Factor Lab method catalog.

This module deliberately has no project-package or network dependency.  It can
validate the frozen registry from a clean research environment without
downloading a method, dataset, or notebook.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class CatalogValidationError(ValueError):
    """Raised when a method catalog cannot be trusted."""


TOP_LEVEL_KEYS = {
    "schema_version",
    "generated_at",
    "source_policy",
    "cards",
    "catalog_sha256",
}
CARD_KEYS = {
    "method_id",
    "version",
    "title",
    "sport",
    "category",
    "status",
    "source",
    "licenses",
    "research",
    "reuse",
    "evidence_boundary",
    "content_sha256",
}
SOURCE_KEYS = {
    "url",
    "source_kind",
    "observed_at",
    "precise_version",
    "primary_source",
    "provenance_notes",
}
LICENSES_KEYS = {"code", "data"}
LICENSE_KEYS = {"status", "spdx_id", "evidence_url", "notes"}
RESEARCH_KEYS = {
    "research_question",
    "inputs",
    "target",
    "validation",
    "published_claims_not_evidence",
}
REUSE_KEYS = {
    "reusable_portion",
    "independently_reproduced",
    "implementation_state",
    "local_dependencies",
    "prohibited_uses",
}
BOUNDARY_KEYS = {"data_gaps", "claim_boundary"}

CARD_STATUSES = {
    "READY_FOR_LOCAL_REPRODUCTION",
    "METHOD_REFERENCE_ONLY",
    "DATA_GAP",
}
LICENSE_STATUSES = {"VERIFIED", "NOT_APPLICABLE", "UNKNOWN"}


def canonical_sha256(value: object) -> str:
    """Return a stable hash for JSON-compatible material."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CatalogValidationError(f"non-canonical JSON value: {exc}") from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogValidationError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CatalogValidationError(
            f"{context} keys invalid; missing={missing}; extra={extra}"
        )
    return value


def _require_nonempty_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogValidationError(f"{context} must be a non-empty string")
    return value


def _require_string_list(value: object, *, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise CatalogValidationError(f"{context} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise CatalogValidationError(
            f"{context} must contain only non-empty strings"
        )
    return value


def _validate_license(value: object, *, context: str) -> dict[str, Any]:
    license_record = _require_exact_keys(value, LICENSE_KEYS, context=context)
    status = _require_nonempty_string(
        license_record["status"], context=f"{context}.status"
    )
    if status not in LICENSE_STATUSES:
        raise CatalogValidationError(f"{context}.status is invalid: {status}")
    _require_nonempty_string(
        license_record["spdx_id"], context=f"{context}.spdx_id"
    )
    evidence_url = _require_nonempty_string(
        license_record["evidence_url"], context=f"{context}.evidence_url"
    )
    if not evidence_url.startswith("https://"):
        raise CatalogValidationError(
            f"{context}.evidence_url must use https"
        )
    _require_nonempty_string(
        license_record["notes"], context=f"{context}.notes"
    )
    return license_record


def _validate_card(value: object, *, index: int) -> dict[str, Any]:
    context = f"cards[{index}]"
    card = _require_exact_keys(value, CARD_KEYS, context=context)
    method_id = _require_nonempty_string(
        card["method_id"], context=f"{context}.method_id"
    )
    if not method_id.startswith("METHOD-"):
        raise CatalogValidationError(
            f"{context}.method_id must start with METHOD-"
        )
    for field in ("version", "title", "sport", "category"):
        _require_nonempty_string(card[field], context=f"{context}.{field}")
    if card["status"] not in CARD_STATUSES:
        raise CatalogValidationError(
            f"{context}.status is invalid: {card['status']}"
        )

    source = _require_exact_keys(
        card["source"], SOURCE_KEYS, context=f"{context}.source"
    )
    for field in (
        "url",
        "source_kind",
        "observed_at",
        "precise_version",
        "provenance_notes",
    ):
        _require_nonempty_string(
            source[field], context=f"{context}.source.{field}"
        )
    if not source["url"].startswith("https://"):
        raise CatalogValidationError(f"{context}.source.url must use https")
    if source["primary_source"] is not True:
        raise CatalogValidationError(
            f"{context}.source.primary_source must be true"
        )

    licenses = _require_exact_keys(
        card["licenses"], LICENSES_KEYS, context=f"{context}.licenses"
    )
    code_license = _validate_license(
        licenses["code"], context=f"{context}.licenses.code"
    )
    data_license = _validate_license(
        licenses["data"], context=f"{context}.licenses.data"
    )
    if card["status"] == "READY_FOR_LOCAL_REPRODUCTION" and (
        code_license["status"] == "UNKNOWN"
        or data_license["status"] == "UNKNOWN"
    ):
        raise CatalogValidationError(
            f"{method_id} cannot be ready while a license is UNKNOWN"
        )

    research = _require_exact_keys(
        card["research"], RESEARCH_KEYS, context=f"{context}.research"
    )
    for field in ("research_question", "target", "validation"):
        _require_nonempty_string(
            research[field], context=f"{context}.research.{field}"
        )
    _require_string_list(
        research["inputs"], context=f"{context}.research.inputs"
    )
    if research["published_claims_not_evidence"] is not True:
        raise CatalogValidationError(
            f"{context}.research.published_claims_not_evidence must be true"
        )

    reuse = _require_exact_keys(
        card["reuse"], REUSE_KEYS, context=f"{context}.reuse"
    )
    _require_string_list(
        reuse["reusable_portion"], context=f"{context}.reuse.reusable_portion"
    )
    _require_string_list(
        reuse["independently_reproduced"],
        context=f"{context}.reuse.independently_reproduced",
    )
    _require_nonempty_string(
        reuse["implementation_state"],
        context=f"{context}.reuse.implementation_state",
    )
    if not isinstance(reuse["local_dependencies"], list) or not all(
        isinstance(item, str) and item.strip()
        for item in reuse["local_dependencies"]
    ):
        raise CatalogValidationError(
            f"{context}.reuse.local_dependencies must be a string list"
        )
    _require_string_list(
        reuse["prohibited_uses"], context=f"{context}.reuse.prohibited_uses"
    )
    if (
        code_license["status"] == "UNKNOWN"
        or data_license["status"] == "UNKNOWN"
    ) and reuse["implementation_state"] != "NOT_INTEGRATED":
        raise CatalogValidationError(
            f"{method_id} has unknown license and must remain NOT_INTEGRATED"
        )

    boundary = _require_exact_keys(
        card["evidence_boundary"],
        BOUNDARY_KEYS,
        context=f"{context}.evidence_boundary",
    )
    _require_string_list(
        boundary["data_gaps"],
        context=f"{context}.evidence_boundary.data_gaps",
    )
    _require_nonempty_string(
        boundary["claim_boundary"],
        context=f"{context}.evidence_boundary.claim_boundary",
    )

    content_hash = _require_nonempty_string(
        card["content_sha256"], context=f"{context}.content_sha256"
    )
    expected_hash = canonical_sha256(
        {key: item for key, item in card.items() if key != "content_sha256"}
    )
    if content_hash != expected_hash:
        raise CatalogValidationError(
            f"{method_id} content_sha256 mismatch: "
            f"expected {expected_hash}, got {content_hash}"
        )
    return card


def validate_method_catalog(value: object) -> dict[str, Any]:
    """Validate a parsed MethodCatalogV1 object and return it unchanged."""

    catalog = _require_exact_keys(
        value, TOP_LEVEL_KEYS, context="method_catalog"
    )
    if catalog["schema_version"] != "MethodCatalogV1":
        raise CatalogValidationError("unsupported schema_version")
    for field in ("generated_at", "source_policy"):
        _require_nonempty_string(
            catalog[field], context=f"method_catalog.{field}"
        )
    cards = catalog["cards"]
    if not isinstance(cards, list) or not cards:
        raise CatalogValidationError("method_catalog.cards must be non-empty")
    validated = [
        _validate_card(card, index=index) for index, card in enumerate(cards)
    ]
    method_ids = [card["method_id"] for card in validated]
    if len(method_ids) != len(set(method_ids)):
        raise CatalogValidationError("duplicate method_id")
    if method_ids != sorted(method_ids):
        raise CatalogValidationError("cards must be sorted by method_id")

    expected_hash = canonical_sha256(
        {
            key: item
            for key, item in catalog.items()
            if key != "catalog_sha256"
        }
    )
    if catalog["catalog_sha256"] != expected_hash:
        raise CatalogValidationError(
            "catalog_sha256 mismatch: "
            f"expected {expected_hash}, got {catalog['catalog_sha256']}"
        )
    return catalog


def load_method_catalog(path: str | Path) -> dict[str, Any]:
    """Load and validate a UTF-8 JSON catalog."""

    catalog_path = Path(path)
    try:
        value = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogValidationError(
            f"cannot read method catalog {catalog_path}: {exc}"
        ) from exc
    return validate_method_catalog(value)

