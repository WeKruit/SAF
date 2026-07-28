"""Build the reaction-blind, moneyline-only NFL factor registry V2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REGISTRY = (
    PROJECT_ROOT / "registries/factors/nfl_factor_registry_v1.json"
)
GOVERNANCE_OBJECT = (
    PROJECT_ROOT
    / "artifacts/market-observation/nfl/x13/factor-lab/v2"
    / "expansion-registry/objects/sha256/22"
    / "226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d"
    ".json"
)
OUTPUT_REGISTRY = (
    PROJECT_ROOT / "registries/factors/nfl_factor_registry_v2.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "registries/factors/nfl_factor_registry_v2"

EXPECTED_SOURCE_SHA256 = (
    "sha256:112fb9cc49d973c160579befdcd1835603b75090e1f6cb1cd39df9b89560567d"
)
EXPECTED_GOVERNANCE_SHA256 = (
    "sha256:226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d"
)
BUILDER_VERSION = "nfl-factor-registry-v2-builder-v1"
REGISTRY_VERSION = "v2"
EXPECTED_FACTOR_COUNT = 104
FORMAL_HORIZONS = (1, 2, 5, 10, 20, 30, 60)
AUTHORIZED_MARKET_FAMILIES = ("moneyline",)
FORMAL_ANALYSIS_ROLE = "FORMAL_ANALYSIS"
DISPLAY_ONLY_ROLE = "DISPLAY_ONLY"
DISPLAY_ONLY_FACTOR_IDS = frozenset(
    {
        "NFL.MARKET.PRIOR_LOW",
        "NFL.MARKET.PRIOR_MID",
        "NFL.MARKET.PRIOR_HIGH",
    }
)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FORBIDDEN_FACTOR_INPUT_TOKENS = frozenset({"price", "size", "direction"})
_PRICE_DERIVED_FACTOR_FIELDS = frozenset({"pre_event_actual_trade"})
_FACTOR_LOCKED_FIELDS = (
    "factor_id",
    "claim_boundary",
    "exclusions",
    "focal_entity",
    "minimum_episodes",
    "minimum_episodes_per_venue",
    "minimum_games",
    "pit_fields",
    "predicate",
    "sports_meaning",
    "support_kind",
    "target",
)
_TEMPLATE_LOCKED_FIELDS = (
    "claim_boundary",
    "event_factor_ids",
    "maximum_order",
    "minimum_episodes",
    "minimum_episodes_per_venue",
    "minimum_games",
    "moderator_prefixes",
    "support_kind",
)


class FactorRegistryV2Error(RuntimeError):
    """The V2 factor-registry build failed closed."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FactorRegistryV2Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_json_number(value: str) -> None:
    raise FactorRegistryV2Error(f"non-JSON numeric literal: {value}")


def _load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FactorRegistryV2Error(f"{label} is not strict JSON") from error
    if type(value) is not dict:
        raise FactorRegistryV2Error(f"{label} must be a JSON object")
    return value


def _read_regular(path: Path, *, label: str, max_bytes: int) -> bytes:
    resolved = path.resolve()
    try:
        before = os.stat(resolved, follow_symlinks=False)
        if resolved.is_symlink() or not resolved.is_file():
            raise OSError("not a regular file")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise OSError("byte length outside bounded input limit")
        with resolved.open("rb") as stream:
            payload = stream.read(max_bytes + 1)
        after = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise FactorRegistryV2Error(
            f"{label} is not a bounded stable regular file: {resolved}"
        ) from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(payload) != before.st_size:
        raise FactorRegistryV2Error(f"{label} changed while being read")
    return payload


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise FactorRegistryV2Error(
            f"path escapes project root: {resolved}"
        ) from error


def _binding(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "path": _repo_path(path),
        "sha256": _sha256(payload),
        "byte_length": len(payload),
    }


def _definition_sha256(definition: dict[str, object]) -> str:
    unhashed = {
        key: value
        for key, value in definition.items()
        if key != "definition_sha256"
    }
    return _sha256(_canonical_bytes(unhashed))


def _locked_projection(
    definition: dict[str, object],
    fields: tuple[str, ...],
) -> dict[str, object]:
    if any(field not in definition for field in fields):
        raise FactorRegistryV2Error("definition lacks a locked semantic field")
    return {field: copy.deepcopy(definition[field]) for field in fields}


def _input_token_hits(definition: dict[str, object]) -> set[str]:
    signal_inputs = {
        "factor_id": definition.get("factor_id"),
        "pit_fields": definition.get("pit_fields"),
        "predicate": definition.get("predicate"),
    }
    text = json.dumps(
        signal_inputs,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).lower()
    tokens = set(filter(None, re.split(r"[^a-z0-9]+", text)))
    hits = tokens & _FORBIDDEN_FACTOR_INPUT_TOKENS
    predicate = definition.get("predicate")
    pit_fields = definition.get("pit_fields")
    referenced_fields = {
        str(row.get("field"))
        for row in predicate
        if type(predicate) is list and type(row) is dict
    } | {
        str(field)
        for field in pit_fields
        if type(pit_fields) is list
    }
    if referenced_fields & _PRICE_DERIVED_FACTOR_FIELDS:
        hits.add("price")
    return hits


def _verify_source_registry(
    path: Path,
) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(
        path,
        label="NFL factor registry V1",
        max_bytes=2 * 1024 * 1024,
    )
    if _sha256(payload) != EXPECTED_SOURCE_SHA256:
        raise FactorRegistryV2Error("V1 registry hash is not authoritative")
    registry = _load_json(payload, label="NFL factor registry V1")
    factors = registry.get("factor_definitions")
    templates = registry.get("interaction_templates")
    if (
        registry.get("schema") != "nfl_factor_registry_v1"
        or registry.get("version") != "v1"
        or type(factors) is not list
        or len(factors) != EXPECTED_FACTOR_COUNT
        or type(templates) is not list
        or not templates
    ):
        raise FactorRegistryV2Error("V1 registry contract is malformed")
    ids = [
        definition.get("factor_id")
        for definition in factors
        if type(definition) is dict
    ]
    if (
        len(ids) != EXPECTED_FACTOR_COUNT
        or any(type(factor_id) is not str for factor_id in ids)
        or len(set(ids)) != EXPECTED_FACTOR_COUNT
    ):
        raise FactorRegistryV2Error("V1 factor IDs are malformed or duplicated")
    return registry, payload


def _verify_governance(
    path: Path,
) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(
        path,
        label="Factor V2 expansion governance",
        max_bytes=2 * 1024 * 1024,
    )
    if _sha256(payload) != EXPECTED_GOVERNANCE_SHA256:
        raise FactorRegistryV2Error("governance hash is not authoritative")
    governance = _load_json(payload, label="Factor V2 expansion governance")
    axes = governance.get("factor_axes")
    probability = axes.get("probability") if type(axes) is dict else None
    timeline = axes.get("timeline") if type(axes) is dict else None
    scope = governance.get("analysis_scope")
    if (
        governance.get("schema") != "nfl_factor_expansion_registry_v2"
        or governance.get("factor_registry_version") != REGISTRY_VERSION
        or governance.get("status") != "FROZEN"
        or type(axes) is not dict
        or axes.get("horizons_seconds") != list(FORMAL_HORIZONS)
        or type(timeline) is not dict
        or type(timeline.get("formal_buckets")) is not list
        or type(probability) is not dict
        or probability.get("formal_representation") != "CONTINUOUS"
        or probability.get("display_labels_are_formal_values") is not False
        or type(probability.get("display_only_bins")) is not list
        or type(scope) is not dict
        or scope.get("authorized_market_families") != ["moneyline"]
        or scope.get("market_value_or_direction_accessed") is not False
    ):
        raise FactorRegistryV2Error("governance axes or scope are malformed")
    return governance, payload


def _build_registry(
    source_registry: dict[str, Any],
    source_bytes: bytes,
    governance: dict[str, Any],
    governance_bytes: bytes,
    *,
    source_path: Path,
    governance_path: Path,
) -> dict[str, object]:
    source_factors = source_registry["factor_definitions"]
    source_templates = source_registry["interaction_templates"]
    assert type(source_factors) is list
    assert type(source_templates) is list

    factors: list[dict[str, object]] = []
    for source_definition in source_factors:
        if type(source_definition) is not dict:
            raise FactorRegistryV2Error("V1 factor definition is malformed")
        before = _locked_projection(
            source_definition,
            _FACTOR_LOCKED_FIELDS,
        )
        definition = copy.deepcopy(source_definition)
        definition["version"] = REGISTRY_VERSION
        definition["horizons_seconds"] = list(FORMAL_HORIZONS)
        definition["market_families"] = list(AUTHORIZED_MARKET_FAMILIES)
        factor_id = str(definition["factor_id"])
        definition["analysis_role"] = (
            DISPLAY_ONLY_ROLE
            if factor_id in DISPLAY_ONLY_FACTOR_IDS
            else FORMAL_ANALYSIS_ROLE
        )
        prohibited_hits = _input_token_hits(definition)
        if (
            _locked_projection(definition, _FACTOR_LOCKED_FIELDS) != before
            or (
                prohibited_hits
                and definition["analysis_role"] != DISPLAY_ONLY_ROLE
            )
        ):
            raise FactorRegistryV2Error(
                f"{definition.get('factor_id')} changed locked semantics "
                "or uses a prohibited market input"
            )
        definition["definition_sha256"] = _definition_sha256(definition)
        factors.append(definition)

    factor_ids = {str(definition["factor_id"]) for definition in factors}
    if (
        {
            str(definition["factor_id"])
            for definition in factors
            if definition["analysis_role"] == DISPLAY_ONLY_ROLE
        }
        != DISPLAY_ONLY_FACTOR_IDS
    ):
        raise FactorRegistryV2Error(
            "probability display-only factor identities differ"
        )
    templates: list[dict[str, object]] = []
    for source_template in source_templates:
        if type(source_template) is not dict:
            raise FactorRegistryV2Error("V1 interaction template is malformed")
        before = _locked_projection(
            source_template,
            _TEMPLATE_LOCKED_FIELDS,
        )
        template = copy.deepcopy(source_template)
        template_id = template.get("template_id")
        if type(template_id) is not str or not template_id.endswith(".V1"):
            raise FactorRegistryV2Error("V1 interaction template ID is malformed")
        template["template_id"] = template_id.removesuffix(".V1") + ".V2"
        template["version"] = REGISTRY_VERSION
        template["horizons_seconds"] = list(FORMAL_HORIZONS)
        template["market_families"] = list(AUTHORIZED_MARKET_FAMILIES)
        if (
            _locked_projection(template, _TEMPLATE_LOCKED_FIELDS) != before
            or template.get("maximum_order") != 2
            or not set(template.get("event_factor_ids", [])) <= factor_ids
        ):
            raise FactorRegistryV2Error(
                "interaction template changed locked support semantics"
            )
        template["definition_sha256"] = _definition_sha256(template)
        templates.append(template)

    interaction_policy = copy.deepcopy(source_registry["interaction_policy"])
    if type(interaction_policy) is not dict:
        raise FactorRegistryV2Error("V1 interaction policy is malformed")
    interaction_policy["maximum_order"] = 2
    interaction_policy["third_order_enabled"] = False
    interaction_policy["eligible_definition_role"] = FORMAL_ANALYSIS_ROLE

    axes = copy.deepcopy(governance["factor_axes"])
    return {
        "schema": "nfl_factor_registry_v2",
        "version": REGISTRY_VERSION,
        "experiment_id": source_registry["experiment_id"],
        "status": source_registry["status"],
        "source_bindings": {
            "factor_registry_v1": _binding(source_path, source_bytes),
            "factor_v2_governance": _binding(
                governance_path,
                governance_bytes,
            ),
        },
        "factor_axes": axes,
        "horizons_seconds": list(FORMAL_HORIZONS),
        "market_scope": {
            "authorized_market_families": list(AUTHORIZED_MARKET_FAMILIES),
            "all_other_market_families": "NOT_AUTHORIZED",
            "formal_factor_inputs_use_price_size_or_direction": False,
        },
        "analysis_roles": {
            "formal": FORMAL_ANALYSIS_ROLE,
            "display_only": DISPLAY_ONLY_ROLE,
            "display_only_factor_ids": sorted(DISPLAY_ONLY_FACTOR_IDS),
            "display_only_definitions_enter_formal_analysis": False,
        },
        "factor_definitions": factors,
        "interaction_policy": interaction_policy,
        "interaction_templates": templates,
        "third_order_policy": "PROHIBITED",
        "target_adapters": {
            "moneyline": {
                "families": list(AUTHORIZED_MARKET_FAMILIES),
                "orientation": "focal_team",
            }
        },
        "bootstrap_samples": source_registry["bootstrap_samples"],
        "bootstrap_seed": source_registry["bootstrap_seed"],
        "historical_microstructure_exclusions": copy.deepcopy(
            source_registry["historical_microstructure_exclusions"]
        ),
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FactorRegistryV2Error(
                f"immutable publication conflicts: {path}"
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _publish(
    registry: dict[str, object],
    *,
    registry_path: Path,
    output_root: Path,
) -> dict[str, object]:
    registry_bytes = _canonical_bytes(registry)
    registry_sha256 = _sha256(registry_bytes)
    registry_hex = registry_sha256.removeprefix("sha256:")
    object_relative = (
        Path("objects")
        / "sha256"
        / registry_hex[:2]
        / f"{registry_hex}.json"
    )
    object_path = output_root / object_relative
    factor_hashes = {
        str(definition["factor_id"]): definition["definition_sha256"]
        for definition in registry["factor_definitions"]
        if type(definition) is dict
    }
    template_hashes = {
        str(template["template_id"]): template["definition_sha256"]
        for template in registry["interaction_templates"]
        if type(template) is dict
    }
    manifest = {
        "schema": "nfl_factor_registry_manifest_v2",
        "builder_version": BUILDER_VERSION,
        "registry_path": _repo_path(registry_path),
        "object_path": object_relative.as_posix(),
        "registry_sha256": registry_sha256,
        "object_sha256": registry_sha256,
        "byte_length": len(registry_bytes),
        "factor_count": len(factor_hashes),
        "formal_factor_count": sum(
            definition.get("analysis_role") == FORMAL_ANALYSIS_ROLE
            for definition in registry["factor_definitions"]
            if type(definition) is dict
        ),
        "display_only_factor_count": sum(
            definition.get("analysis_role") == DISPLAY_ONLY_ROLE
            for definition in registry["factor_definitions"]
            if type(definition) is dict
        ),
        "interaction_template_count": len(template_hashes),
        "factor_definition_sha256": factor_hashes,
        "interaction_template_definition_sha256": template_hashes,
        "source_registry_sha256": EXPECTED_SOURCE_SHA256,
        "governance_object_sha256": EXPECTED_GOVERNANCE_SHA256,
    }
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha256 = _sha256(manifest_bytes)
    manifest_hex = manifest_sha256.removeprefix("sha256:")
    manifest_relative = (
        Path("manifests") / f"{manifest_hex}.manifest.json"
    )
    manifest_path = output_root / manifest_relative
    _atomic_write(registry_path, registry_bytes)
    _atomic_write(object_path, registry_bytes)
    _atomic_write(manifest_path, manifest_bytes)
    return {
        **manifest,
        "manifest_path": manifest_relative.as_posix(),
        "manifest_sha256": manifest_sha256,
    }


def _bounded_validate(
    source_registry: dict[str, Any],
    registry: dict[str, object],
    publication: dict[str, object],
    *,
    registry_path: Path,
    output_root: Path,
) -> list[str]:
    checks: list[str] = []
    source_factors = source_registry.get("factor_definitions")
    factors = registry.get("factor_definitions")
    templates = registry.get("interaction_templates")
    if (
        type(source_factors) is not list
        or type(factors) is not list
        or type(templates) is not list
    ):
        raise FactorRegistryV2Error("bounded validation: definitions missing")
    source_by_id = {
        str(definition["factor_id"]): definition
        for definition in source_factors
        if type(definition) is dict
    }
    factor_by_id = {
        str(definition["factor_id"]): definition
        for definition in factors
        if type(definition) is dict
    }
    if (
        len(source_by_id) != EXPECTED_FACTOR_COUNT
        or len(factor_by_id) != EXPECTED_FACTOR_COUNT
        or set(factor_by_id) != set(source_by_id)
    ):
        raise FactorRegistryV2Error(
            "bounded validation: 104 factor IDs are not identical"
        )
    checks.append("104_FACTOR_IDS_IDENTICAL_AND_UNIQUE")

    for factor_id, definition in factor_by_id.items():
        source = source_by_id[factor_id]
        expected_role = (
            DISPLAY_ONLY_ROLE
            if factor_id in DISPLAY_ONLY_FACTOR_IDS
            else FORMAL_ANALYSIS_ROLE
        )
        if (
            definition.get("version") != REGISTRY_VERSION
            or definition.get("horizons_seconds") != list(FORMAL_HORIZONS)
            or definition.get("market_families") != ["moneyline"]
            or definition.get("analysis_role") != expected_role
            or _locked_projection(definition, _FACTOR_LOCKED_FIELDS)
            != _locked_projection(source, _FACTOR_LOCKED_FIELDS)
            or (
                _input_token_hits(definition)
                and expected_role != DISPLAY_ONLY_ROLE
            )
            or definition.get("definition_sha256")
            != _definition_sha256(definition)
        ):
            raise FactorRegistryV2Error(
                f"bounded validation: invalid factor {factor_id}"
            )
    checks.append("FACTOR_SEMANTICS_UNCHANGED_AND_HASHED")

    if (
        registry.get("analysis_roles")
        != {
            "formal": FORMAL_ANALYSIS_ROLE,
            "display_only": DISPLAY_ONLY_ROLE,
            "display_only_factor_ids": sorted(DISPLAY_ONLY_FACTOR_IDS),
            "display_only_definitions_enter_formal_analysis": False,
        }
        or sum(
            definition.get("analysis_role") == FORMAL_ANALYSIS_ROLE
            for definition in factors
            if type(definition) is dict
        )
        != EXPECTED_FACTOR_COUNT - len(DISPLAY_ONLY_FACTOR_IDS)
        or sum(
            definition.get("analysis_role") == DISPLAY_ONLY_ROLE
            for definition in factors
            if type(definition) is dict
        )
        != len(DISPLAY_ONLY_FACTOR_IDS)
    ):
        raise FactorRegistryV2Error(
            "bounded validation: probability bins are not display-only"
        )
    checks.append("CONTINUOUS_PROBABILITY_101_FORMAL_3_DISPLAY_ONLY")

    if any(
        template.get("version") != REGISTRY_VERSION
        or template.get("horizons_seconds") != list(FORMAL_HORIZONS)
        or template.get("market_families") != ["moneyline"]
        or template.get("maximum_order") != 2
        or template.get("definition_sha256")
        != _definition_sha256(template)
        for template in templates
        if type(template) is dict
    ):
        raise FactorRegistryV2Error(
            "bounded validation: invalid interaction template"
        )
    checks.append("INTERACTION_TEMPLATES_V2_HASHED_AND_ORDER_TWO")

    all_horizons = {
        horizon
        for definition in [*factors, *templates]
        if type(definition) is dict
        for horizon in definition.get("horizons_seconds", [])
    }
    if (
        registry.get("horizons_seconds") != list(FORMAL_HORIZONS)
        or all_horizons != set(FORMAL_HORIZONS)
    ):
        raise FactorRegistryV2Error(
            "bounded validation: unregistered horizon"
        )
    checks.append("ONLY_REGISTERED_HORIZONS_1_2_5_10_20_30_60")

    governance_axes = registry.get("factor_axes")
    probability = (
        governance_axes.get("probability")
        if type(governance_axes) is dict
        else None
    )
    if (
        registry.get("third_order_policy") != "PROHIBITED"
        or registry.get("market_scope")
        != {
            "authorized_market_families": ["moneyline"],
            "all_other_market_families": "NOT_AUTHORIZED",
            "formal_factor_inputs_use_price_size_or_direction": False,
        }
        or registry.get("interaction_policy", {}).get(
            "eligible_definition_role"
        )
        != FORMAL_ANALYSIS_ROLE
        or type(probability) is not dict
        or probability.get("formal_representation") != "CONTINUOUS"
        or probability.get("display_labels_are_formal_values") is not False
    ):
        raise FactorRegistryV2Error(
            "bounded validation: governance scope mismatch"
        )
    checks.append("MONEYLINE_CONTINUOUS_PROBABILITY_NO_THIRD_ORDER")

    registry_bytes = _read_regular(
        registry_path,
        label="published V2 registry",
        max_bytes=2 * 1024 * 1024,
    )
    object_relative = publication.get("object_path")
    manifest_relative = publication.get("manifest_path")
    if type(object_relative) is not str or type(manifest_relative) is not str:
        raise FactorRegistryV2Error(
            "bounded validation: publication paths malformed"
        )
    object_bytes = _read_regular(
        output_root / object_relative,
        label="published V2 registry object",
        max_bytes=2 * 1024 * 1024,
    )
    manifest_bytes = _read_regular(
        output_root / manifest_relative,
        label="published V2 registry manifest",
        max_bytes=2 * 1024 * 1024,
    )
    if (
        registry_bytes != _canonical_bytes(registry)
        or object_bytes != registry_bytes
        or _sha256(registry_bytes) != publication.get("registry_sha256")
        or _sha256(manifest_bytes) != publication.get("manifest_sha256")
    ):
        raise FactorRegistryV2Error(
            "bounded validation: content-address round trip failed"
        )
    checks.append("REGISTRY_AND_MANIFEST_CONTENT_ADDRESS_ROUND_TRIP")

    repeated = _publish(
        registry,
        registry_path=registry_path,
        output_root=output_root,
    )
    if repeated != publication:
        raise FactorRegistryV2Error(
            "bounded validation: nondeterministic publication"
        )
    checks.append("IDEMPOTENT_DETERMINISTIC_REPUBLISH")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the frozen NFL factor registry V2"
    )
    parser.add_argument("--source-registry", type=Path, default=SOURCE_REGISTRY)
    parser.add_argument("--governance-object", type=Path, default=GOVERNANCE_OBJECT)
    parser.add_argument("--output-registry", type=Path, default=OUTPUT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bounded-validation", action="store_true")
    args = parser.parse_args()

    source_registry, source_bytes = _verify_source_registry(
        args.source_registry.resolve()
    )
    governance, governance_bytes = _verify_governance(
        args.governance_object.resolve()
    )
    registry = _build_registry(
        source_registry,
        source_bytes,
        governance,
        governance_bytes,
        source_path=args.source_registry.resolve(),
        governance_path=args.governance_object.resolve(),
    )
    publication = _publish(
        registry,
        registry_path=args.output_registry.resolve(),
        output_root=args.output_root.resolve(),
    )
    checks = (
        _bounded_validate(
            source_registry,
            registry,
            publication,
            registry_path=args.output_registry.resolve(),
            output_root=args.output_root.resolve(),
        )
        if args.bounded_validation
        else []
    )
    print(
        json.dumps(
            {
                **publication,
                "bounded_validation": {
                    "status": "PASS" if args.bounded_validation else "NOT_REQUESTED",
                    "checks": checks,
                },
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
