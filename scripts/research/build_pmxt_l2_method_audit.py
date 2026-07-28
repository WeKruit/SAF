#!/usr/bin/env python3
"""Create a PMXT L2 method audit from local fixtures and metadata only."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prediction_market.pmxt.microstructure_metrics import (  # noqa: E402
    L2MethodError,
    build_method_audit_payload,
    method_audit_fixture_events,
)


def _json_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise L2MethodError(f"cannot read local JSON metadata: {path}") from exc
    if not isinstance(value, Mapping):
        raise L2MethodError(f"local JSON metadata must be an object: {path}")
    return value


def _jsonl_events(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise L2MethodError(f"cannot read local JSONL fixture: {path}") from exc
    events: list[Mapping[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise L2MethodError(f"fixture JSONL line {line_number} is invalid") from exc
        if not isinstance(event, Mapping):
            raise L2MethodError(f"fixture JSONL line {line_number} must be an object")
        events.append(event)
    if not events:
        raise L2MethodError("fixture JSONL contains no events")
    return tuple(events)


def _manifest_records(root: Path) -> tuple[Mapping[str, object], ...]:
    if not root.is_dir():
        raise L2MethodError(f"local manifest root does not exist: {root}")
    paths = tuple(sorted(root.rglob("*.manifest.json")))
    if not paths:
        raise L2MethodError(f"local manifest root contains no manifests: {root}")
    return tuple(_json_object(path) for path in paths)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        help="local frozen PMXT JSONL fixture; defaults to the built-in gap/reconnect fixture",
    )
    parser.add_argument(
        "--remote-inventory",
        type=Path,
        required=True,
        help="local PMXT remote-verification JSON inventory",
    )
    parser.add_argument(
        "--phase0-inventory",
        type=Path,
        default=PROJECT_ROOT / "artifacts/data-audit/phase0_inventory.json",
        help="local Phase-0 PMXT inventory JSON",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=PROJECT_ROOT / "var/raw/manifests/source=pmxt/dataset=DS-PMXT-V2/version=v2",
        help="local authoritative PMXT manifest directory",
    )
    parser.add_argument(
        "--phase1-measurement",
        type=Path,
        help="local frozen Phase-1 sample measurement JSON for non-manifest evidence",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination JSON path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    phase1_measurement = (
        _json_object(args.phase1_measurement)
        if args.phase1_measurement is not None
        else None
    )
    phase1_sha256 = (
        "sha256:"
        + hashlib.sha256(args.phase1_measurement.read_bytes()).hexdigest()
        if args.phase1_measurement is not None
        else None
    )
    audit = build_method_audit_payload(
        fixture_events=(
            _jsonl_events(args.fixture)
            if args.fixture is not None
            else method_audit_fixture_events()
        ),
        remote_inventory=_json_object(args.remote_inventory),
        phase0_inventory=_json_object(args.phase0_inventory),
        authoritative_manifests=_manifest_records(args.manifest_root),
        phase1_sample_measurement=phase1_measurement,
        phase1_measurement_sha256=phase1_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except L2MethodError as exc:
        raise SystemExit(f"PMXT L2 method audit failed: {exc}") from exc
