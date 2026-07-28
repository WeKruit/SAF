"""Open NFL expansion development reaction access after every frozen gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from prediction_market.sports.nfl_historical_moneyline_capture import (
    HistoricalMoneylineCaptureError,
    dry_run_moneyline_expansion,
    load_moneyline_expansion_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SCAN = (
    PROJECT_ROOT
    / "artifacts/market-observation/nfl/x13/candidate-coverage"
    / "nfl_2025_dual_venue_candidate_scan_v0.json"
)
TEMPORAL_AUDIT = (
    PROJECT_ROOT
    / "artifacts/market-observation/nfl/x13/candidate-coverage"
    / "temporal-all-dual-v1/objects/sha256/43"
    / "432277924ccc8249333b48b96060662cf630bf1a72616e79d9e186bdb2de1588"
    ".json"
)
GOVERNANCE_OBJECT = (
    PROJECT_ROOT
    / "artifacts/market-observation/nfl/x13/factor-lab/v2"
    / "expansion-registry/objects/sha256/22"
    / "226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d"
    ".json"
)
FACTOR_REGISTRY = (
    PROJECT_ROOT / "registries/factors/nfl_factor_registry_v2.json"
)
SPORTS_BATCH = (
    PROJECT_ROOT
    / "artifacts/market-observation/nfl/x13/sports-clean-expansion"
    / "reaction-closed-v1/batches/manifests/sha256/3f"
    / "3fee02261ae6aeeccbfaa03114ebdbdfd44e96eb7dad1501de5f39c2bce0271b"
    ".batch-index.json"
)
CAPTURE_ROOT = (
    PROJECT_ROOT
    / "artifacts/market-observation/nfl/x13/expansion-capture"
    / "moneyline-reaction-closed-v1"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts/market-observation/nfl/x13/factor-lab/v2"
    / "reaction-access-amendment/development-open-v1"
)

EXPECTED_GOVERNANCE_SHA256 = (
    "sha256:226b796358426185609cd3c6f18f5ab67828d465f194f5403a56a397ed77493d"
)
EXPECTED_FACTOR_REGISTRY_SHA256 = (
    "sha256:295e4e48237f8523fe3153f28b5b28adc9f6e1d3a219568047da5c250ed243c1"
)
EXPECTED_SPORTS_BATCH_SHA256 = (
    "sha256:4ce0759112084e45a7670c4436e240568215bc825c3133c40827061094d2c934"
)
EXPECTED_SPORTS_BATCH_FILE_SHA256 = (
    "sha256:3fee02261ae6aeeccbfaa03114ebdbdfd44e96eb7dad1501de5f39c2bce0271b"
)
EXPECTED_GAME_COUNT = 234
EXPECTED_DEVELOPMENT_COUNT = 153
EXPECTED_FINAL_HOLDOUT_COUNT = 81
EXPECTED_FACTOR_COUNT = 104
EXPECTED_HORIZONS = (1, 2, 5, 10, 20, 30, 60)
BUILDER_VERSION = "nfl-expansion-development-unlock-v1"


class UnlockGateError(RuntimeError):
    """The reaction-access amendment gate failed closed."""

    def __init__(self, reason: str, details: dict[str, object] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def _default_raw_root() -> Path:
    if PROJECT_ROOT.parent.name == ".worktrees":
        return PROJECT_ROOT.parent.parent / "var/raw"
    return PROJECT_ROOT / "var/raw"


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
            raise UnlockGateError("DUPLICATE_JSON_KEY", {"key": key})
        result[key] = value
    return result


def _reject_non_json_number(value: str) -> None:
    raise UnlockGateError("NON_JSON_NUMBER", {"value": value})


def _load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UnlockGateError("INVALID_JSON", {"label": label}) from error
    if type(value) is not dict:
        raise UnlockGateError("INVALID_JSON_ROOT", {"label": label})
    return value


def _read_regular(path: Path, *, label: str, max_bytes: int) -> bytes:
    resolved = path.resolve()
    try:
        before = os.stat(resolved, follow_symlinks=False)
        if resolved.is_symlink() or not resolved.is_file():
            raise OSError("not a regular file")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise OSError("byte length outside bounded limit")
        with resolved.open("rb") as stream:
            payload = stream.read(max_bytes + 1)
        after = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise UnlockGateError(
            "INPUT_NOT_STABLE_REGULAR_FILE",
            {"label": label, "path": str(resolved)},
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
        raise UnlockGateError("INPUT_CHANGED_WHILE_READING", {"label": label})
    return payload


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise UnlockGateError(
            "PATH_ESCAPES_PROJECT_ROOT",
            {"path": str(resolved)},
        ) from error


def _binding(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "path": _repo_path(path),
        "sha256": _sha256(payload),
        "byte_length": len(payload),
    }


def _verify_governance(
    path: Path,
) -> tuple[dict[str, Any], bytes, tuple[str, ...], tuple[str, ...]]:
    payload = _read_regular(
        path,
        label="governance object",
        max_bytes=2 * 1024 * 1024,
    )
    if _sha256(payload) != EXPECTED_GOVERNANCE_SHA256:
        raise UnlockGateError("GOVERNANCE_HASH_MISMATCH")
    governance = _load_json(payload, label="governance object")
    split = governance.get("split_lock")
    scope = governance.get("analysis_scope")
    axes = governance.get("factor_axes")
    assignments = split.get("game_assignments") if type(split) is dict else None
    if (
        governance.get("schema") != "nfl_factor_expansion_registry_v2"
        or governance.get("status") != "FROZEN"
        or type(split) is not dict
        or type(scope) is not dict
        or type(axes) is not dict
        or type(assignments) is not list
        or len(assignments) != EXPECTED_GAME_COUNT
        or scope.get("authorized_market_families") != ["moneyline"]
        or axes.get("horizons_seconds") != list(EXPECTED_HORIZONS)
    ):
        raise UnlockGateError("GOVERNANCE_CONTRACT_MISMATCH")
    development_ids = tuple(
        sorted(
            str(row.get("game_id"))
            for row in assignments
            if type(row) is dict and row.get("cohort") == "development"
        )
    )
    final_holdout_ids = tuple(
        sorted(
            str(row.get("game_id"))
            for row in assignments
            if type(row) is dict and row.get("cohort") == "final_holdout"
        )
    )
    if (
        len(development_ids) != EXPECTED_DEVELOPMENT_COUNT
        or len(final_holdout_ids) != EXPECTED_FINAL_HOLDOUT_COUNT
        or len(set(development_ids) | set(final_holdout_ids))
        != EXPECTED_GAME_COUNT
        or set(development_ids) & set(final_holdout_ids)
    ):
        raise UnlockGateError("GOVERNANCE_SPLIT_MISMATCH")
    return governance, payload, development_ids, final_holdout_ids


def _verify_factor_registry(
    path: Path,
    governance_axes: object,
) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(
        path,
        label="factor registry V2",
        max_bytes=2 * 1024 * 1024,
    )
    if _sha256(payload) != EXPECTED_FACTOR_REGISTRY_SHA256:
        raise UnlockGateError("FACTOR_REGISTRY_HASH_MISMATCH")
    registry = _load_json(payload, label="factor registry V2")
    factors = registry.get("factor_definitions")
    templates = registry.get("interaction_templates")
    market_scope = registry.get("market_scope")
    if (
        registry.get("schema") != "nfl_factor_registry_v2"
        or registry.get("version") != "v2"
        or registry.get("factor_axes") != governance_axes
        or registry.get("horizons_seconds") != list(EXPECTED_HORIZONS)
        or registry.get("third_order_policy") != "PROHIBITED"
        or type(market_scope) is not dict
        or market_scope.get("authorized_market_families") != ["moneyline"]
        or type(factors) is not list
        or len(factors) != EXPECTED_FACTOR_COUNT
        or type(templates) is not list
    ):
        raise UnlockGateError("FACTOR_REGISTRY_CONTRACT_MISMATCH")
    factor_ids = [
        row.get("factor_id")
        for row in factors
        if type(row) is dict
    ]
    if (
        len(factor_ids) != EXPECTED_FACTOR_COUNT
        or any(type(factor_id) is not str for factor_id in factor_ids)
        or len(set(factor_ids)) != EXPECTED_FACTOR_COUNT
        or any(
            type(row) is not dict
            or row.get("version") != "v2"
            or row.get("horizons_seconds") != list(EXPECTED_HORIZONS)
            or row.get("market_families") != ["moneyline"]
            for row in factors
        )
        or any(
            type(row) is not dict
            or row.get("version") != "v2"
            or row.get("horizons_seconds") != list(EXPECTED_HORIZONS)
            or row.get("market_families") != ["moneyline"]
            or row.get("maximum_order") != 2
            for row in templates
        )
    ):
        raise UnlockGateError("FACTOR_DEFINITION_SCOPE_MISMATCH")
    return registry, payload


def _verify_sports_batch(
    path: Path,
    expected_game_ids: set[str],
) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(
        path,
        label="sports batch",
        max_bytes=2 * 1024 * 1024,
    )
    if _sha256(payload) != EXPECTED_SPORTS_BATCH_FILE_SHA256:
        raise UnlockGateError("SPORTS_BATCH_FILE_HASH_MISMATCH")
    batch = _load_json(payload, label="sports batch")
    games = batch.get("games")
    if (
        batch.get("schema") != "nfl_historical_sports_clean_batch_index_v1"
        or batch.get("batch_sha256") != EXPECTED_SPORTS_BATCH_SHA256
        or batch.get("source_scope") != "sports_only"
        or batch.get("reaction_access") != "CLOSED"
        or batch.get("requested_game_count") != EXPECTED_GAME_COUNT
        or batch.get("published_game_count") != EXPECTED_GAME_COUNT
        or batch.get("failed_game_count") != 0
        or batch.get("publication_gate") != "PASS"
        or type(games) is not list
        or len(games) != EXPECTED_GAME_COUNT
    ):
        raise UnlockGateError("SPORTS_BATCH_CONTRACT_MISMATCH")
    game_ids = {
        str(row.get("game_id"))
        for row in games
        if type(row) is dict
    }
    if game_ids != expected_game_ids or len(game_ids) != EXPECTED_GAME_COUNT:
        raise UnlockGateError("SPORTS_BATCH_GAME_SCOPE_MISMATCH")
    return batch, payload


def _verify_capture_batch(
    capture_root: Path,
    spec: object,
    expected_game_ids: set[str],
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    pointer_path = capture_root / "batch/latest.json"
    pointer_bytes = _read_regular(
        pointer_path,
        label="moneyline capture batch pointer",
        max_bytes=64 * 1024,
    )
    pointer = _load_json(
        pointer_bytes,
        label="moneyline capture batch pointer",
    )
    manifest_relative = pointer.get("manifest_path")
    if (
        pointer.get("schema") != "nfl_moneyline_expansion_batch_pointer_v1"
        or pointer.get("batch_plan_sha256") != spec.batch_plan_sha256
        or type(manifest_relative) is not str
        or type(pointer.get("manifest_sha256")) is not str
    ):
        raise UnlockGateError("CAPTURE_BATCH_POINTER_MISMATCH")
    manifest_path = (capture_root / manifest_relative).resolve()
    try:
        manifest_path.relative_to(capture_root.resolve())
    except ValueError as error:
        raise UnlockGateError("CAPTURE_MANIFEST_PATH_ESCAPES_ROOT") from error
    manifest_bytes = _read_regular(
        manifest_path,
        label="moneyline capture batch manifest",
        max_bytes=2 * 1024 * 1024,
    )
    if _sha256(manifest_bytes) != pointer["manifest_sha256"]:
        raise UnlockGateError("CAPTURE_MANIFEST_HASH_MISMATCH")
    manifest = _load_json(
        manifest_bytes,
        label="moneyline capture batch manifest",
    )
    indexes = manifest.get("game_indexes")
    if (
        manifest.get("schema") != "nfl_moneyline_expansion_batch_manifest_v1"
        or manifest.get("batch_plan_sha256") != spec.batch_plan_sha256
        or manifest.get("governance_sha256") != EXPECTED_GOVERNANCE_SHA256
        or manifest.get("capture_scope") != "moneyline-only"
        or manifest.get("reaction_access") != "CLOSED"
        or manifest.get("game_count") != EXPECTED_GAME_COUNT
        or manifest.get("validation_fields")
        != ["identity", "timestamp", "count", "cursor"]
        or type(indexes) is not list
        or len(indexes) != EXPECTED_GAME_COUNT
    ):
        raise UnlockGateError("CAPTURE_BATCH_MANIFEST_MISMATCH")
    index_ids = {
        str(row.get("game_id"))
        for row in indexes
        if type(row) is dict
    }
    if index_ids != expected_game_ids or len(index_ids) != EXPECTED_GAME_COUNT:
        raise UnlockGateError("CAPTURE_BATCH_GAME_SCOPE_MISMATCH")
    for row in indexes:
        if type(row) is not dict:
            raise UnlockGateError("CAPTURE_BATCH_INDEX_ROW_MALFORMED")
        game_id = str(row["game_id"])
        checkpoint_path = capture_root / "checkpoints" / f"{game_id}.json"
        checkpoint_bytes = _read_regular(
            checkpoint_path,
            label=f"{game_id} capture checkpoint",
            max_bytes=64 * 1024,
        )
        checkpoint = _load_json(
            checkpoint_bytes,
            label=f"{game_id} capture checkpoint",
        )
        if (
            checkpoint.get("index_path") != row.get("index_path")
            or checkpoint.get("index_sha256") != row.get("index_sha256")
        ):
            raise UnlockGateError(
                "CAPTURE_BATCH_CHECKPOINT_BINDING_MISMATCH",
                {"game_id": game_id},
            )
    return pointer, pointer_bytes, manifest, manifest_bytes


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise UnlockGateError(
                "CONTENT_ADDRESS_CONFLICT",
                {"path": str(path)},
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
    amendment: dict[str, object],
    *,
    output_root: Path,
) -> dict[str, object]:
    object_bytes = _canonical_bytes(amendment)
    object_sha256 = _sha256(object_bytes)
    object_hex = object_sha256.removeprefix("sha256:")
    object_relative = (
        Path("objects")
        / "sha256"
        / object_hex[:2]
        / f"{object_hex}.json"
    )
    object_path = output_root / object_relative
    manifest = {
        "schema": "nfl_expansion_reaction_access_amendment_manifest_v1",
        "builder_version": BUILDER_VERSION,
        "object_path": object_relative.as_posix(),
        "object_sha256": object_sha256,
        "byte_length": len(object_bytes),
        "development_reaction_access": "OPEN_FOR_REGISTERED_ANALYSIS",
        "final_holdout_reaction_access": "CLOSED",
        "factor_registry_sha256": EXPECTED_FACTOR_REGISTRY_SHA256,
        "governance_object_sha256": EXPECTED_GOVERNANCE_SHA256,
        "sports_batch_sha256": EXPECTED_SPORTS_BATCH_SHA256,
    }
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha256 = _sha256(manifest_bytes)
    manifest_hex = manifest_sha256.removeprefix("sha256:")
    manifest_relative = (
        Path("manifests") / f"{manifest_hex}.manifest.json"
    )
    _atomic_write(object_path, object_bytes)
    _atomic_write(output_root / manifest_relative, manifest_bytes)
    return {
        **manifest,
        "manifest_path": manifest_relative.as_posix(),
        "manifest_sha256": manifest_sha256,
    }


def _evaluate(
    *,
    candidate_scan: Path,
    temporal_audit: Path,
    governance_path: Path,
    factor_registry_path: Path,
    sports_batch_path: Path,
    capture_root: Path,
    raw_root: Path,
    program_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    (
        governance,
        governance_bytes,
        development_ids,
        final_holdout_ids,
    ) = _verify_governance(governance_path)
    factor_registry, factor_registry_bytes = _verify_factor_registry(
        factor_registry_path,
        governance["factor_axes"],
    )
    expected_game_ids = set(development_ids) | set(final_holdout_ids)
    sports_batch, sports_batch_bytes = _verify_sports_batch(
        sports_batch_path,
        expected_game_ids,
    )
    del factor_registry, sports_batch

    try:
        spec = load_moneyline_expansion_spec(
            candidate_scan,
            temporal_audit,
            governance_path,
        )
        capture_audit = dry_run_moneyline_expansion(
            spec,
            raw_root=raw_root,
            program_root=program_root,
            output_root=capture_root,
        )
    except HistoricalMoneylineCaptureError as error:
        raise UnlockGateError(
            "CAPTURE_DRY_RUN_VERIFICATION_FAILED",
            {"error_type": type(error).__name__},
        ) from error
    pending_count = capture_audit.get("pending_game_count")
    resumed_count = capture_audit.get("resumed_game_count")
    if (
        capture_audit.get("schema") != "nfl_moneyline_expansion_dry_run_v1"
        or capture_audit.get("game_count") != EXPECTED_GAME_COUNT
        or capture_audit.get("reaction_access") != "CLOSED"
        or capture_audit.get("network_accessed") is not False
        or type(pending_count) is not int
        or type(resumed_count) is not int
    ):
        raise UnlockGateError("CAPTURE_DRY_RUN_CONTRACT_MISMATCH")
    if pending_count != 0 or resumed_count != EXPECTED_GAME_COUNT:
        raise UnlockGateError(
            "MONEYLINE_CAPTURE_INCOMPLETE",
            {
                "expected_game_count": EXPECTED_GAME_COUNT,
                "pending_game_count": pending_count,
                "verified_completed_game_count": resumed_count,
            },
        )

    (
        capture_pointer,
        capture_pointer_bytes,
        capture_manifest,
        capture_manifest_bytes,
    ) = _verify_capture_batch(capture_root, spec, expected_game_ids)
    del capture_pointer, capture_manifest

    amendment = {
        "schema": "nfl_expansion_reaction_access_amendment_v1",
        "builder_version": BUILDER_VERSION,
        "experiment_id": "X-13",
        "status": "FROZEN",
        "source_bindings": {
            "governance_object": _binding(
                governance_path,
                governance_bytes,
            ),
            "factor_registry_v2": _binding(
                factor_registry_path,
                factor_registry_bytes,
            ),
            "sports_batch": {
                **_binding(sports_batch_path, sports_batch_bytes),
                "batch_sha256": EXPECTED_SPORTS_BATCH_SHA256,
            },
            "moneyline_capture_batch_pointer": _binding(
                capture_root / "batch/latest.json",
                capture_pointer_bytes,
            ),
            "moneyline_capture_batch_manifest": {
                "sha256": _sha256(capture_manifest_bytes),
                "byte_length": len(capture_manifest_bytes),
                "batch_plan_sha256": spec.batch_plan_sha256,
            },
        },
        "split_access": {
            "development": {
                "game_count": len(development_ids),
                "game_ids": list(development_ids),
                "reaction_access": "OPEN_FOR_REGISTERED_ANALYSIS",
            },
            "final_holdout": {
                "game_count": len(final_holdout_ids),
                "game_ids": list(final_holdout_ids),
                "reaction_access": "CLOSED",
            },
        },
        "registered_analysis_authorization": {
            "market_families": ["moneyline"],
            "factor_registry_sha256": EXPECTED_FACTOR_REGISTRY_SHA256,
            "factor_definition_count": EXPECTED_FACTOR_COUNT,
            "horizons_seconds": list(EXPECTED_HORIZONS),
            "factor_axes": governance["factor_axes"],
            "third_order_interactions": "PROHIBITED",
        },
        "capture_verification": {
            "exact_game_count": EXPECTED_GAME_COUNT,
            "raw_manifests_hash_verified": True,
            "request_identity_verified": True,
            "pagination_terminal_proof_verified": True,
            "network_accessed": False,
        },
        "claim_boundary": {
            "development_only": True,
            "final_holdout_remains_closed": True,
            "registered_analysis_only": True,
            "market_value_fields_accessed_by_unlock_gate": False,
        },
    }
    readiness = {
        "status": "READY_TO_PUBLISH",
        "development_game_count": len(development_ids),
        "final_holdout_game_count": len(final_holdout_ids),
        "verified_capture_game_count": resumed_count,
        "factor_definition_count": EXPECTED_FACTOR_COUNT,
        "sports_batch_game_count": EXPECTED_GAME_COUNT,
    }
    return amendment, readiness


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unlock registered development analysis after frozen gates pass"
    )
    parser.add_argument("--candidate-scan", type=Path, default=CANDIDATE_SCAN)
    parser.add_argument("--temporal-audit", type=Path, default=TEMPORAL_AUDIT)
    parser.add_argument("--governance-object", type=Path, default=GOVERNANCE_OBJECT)
    parser.add_argument("--factor-registry", type=Path, default=FACTOR_REGISTRY)
    parser.add_argument("--sports-batch", type=Path, default=SPORTS_BATCH)
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--raw-root", type=Path, default=_default_raw_root())
    parser.add_argument("--program-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        amendment, readiness = _evaluate(
            candidate_scan=args.candidate_scan.resolve(),
            temporal_audit=args.temporal_audit.resolve(),
            governance_path=args.governance_object.resolve(),
            factor_registry_path=args.factor_registry.resolve(),
            sports_batch_path=args.sports_batch.resolve(),
            capture_root=args.capture_root.resolve(),
            raw_root=args.raw_root.resolve(),
            program_root=args.program_root.resolve(),
        )
        if args.dry_run:
            result = {**readiness, "published": False}
        else:
            publication = _publish(
                amendment,
                output_root=args.output_root.resolve(),
            )
            result = {
                **readiness,
                "published": True,
                "publication": publication,
            }
    except UnlockGateError as error:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "reason": error.reason,
                    "details": error.details,
                    "published": False,
                },
                sort_keys=True,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
