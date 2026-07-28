#!/usr/bin/env python3
"""Render the verified exact-153 X-13 development report without upstream I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pyarrow.parquet as pq

from prediction_market.reports.nfl_x13_report import (
    build_report_state,
    render_exact153_html,
    render_exact153_markdown,
)


DEFAULT_ROOT = Path(
    "artifacts/market-observation/nfl/x13/factor-lab/v2/"
    "expansion-development-market/primary-delay0-v1"
)
DEFAULT_DISCOVERY_MANIFEST = Path(
    "cross-game/discovery/manifests/sha256/07/"
    "07efd4809fbbda3da9293f3048d5af28c25ceb45b37ad6bcc4412e929a3b9809"
    ".manifest.json"
)
DEFAULT_OUTPUT = Path("reports/nfl-factor-lab/exact-153")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stage_totals(root: Path, batch: dict[str, object]) -> dict[str, int]:
    totals = {
        "actual_market_observations": 0,
        "reaction_paths": 0,
        "factor_observations": 0,
    }
    games = batch.get("games")
    if not isinstance(games, list) or len(games) != 153:
        raise ValueError("batch must contain the exact 153 games")
    for descriptor in games:
        if not isinstance(descriptor, dict):
            raise ValueError("batch game descriptor is invalid")
        relative = descriptor.get("manifest_path")
        if not isinstance(relative, str):
            raise ValueError("batch game manifest path is invalid")
        manifest_path = root / relative
        expected_sha = descriptor.get("manifest_sha256")
        if _sha256(manifest_path) != expected_sha:
            raise ValueError(f"single-game manifest hash differs: {relative}")
        manifest = _read_json(manifest_path)
        stages = manifest.get("stages")
        if not isinstance(stages, dict):
            raise ValueError("single-game stages are invalid")
        for stage in totals:
            stage_value = stages.get(stage)
            if not isinstance(stage_value, dict):
                raise ValueError(f"single-game stage is missing: {stage}")
            row_count = stage_value.get("row_count")
            if type(row_count) is not int or row_count < 0:
                raise ValueError(f"single-game stage row count is invalid: {stage}")
            totals[stage] += row_count
    return totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--discovery-manifest",
        type=Path,
        default=DEFAULT_DISCOVERY_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    manifest_path = root / args.discovery_manifest
    manifest = _read_json(manifest_path)
    if (
        manifest.get("publication_gate") != "PASS"
        or manifest.get("final_holdout_access") != "CLOSED"
        or manifest.get("cohort") != "development"
    ):
        raise ValueError("discovery manifest is not sealed development evidence")
    discovery_descriptor = manifest.get("discovery")
    batch_descriptor = manifest.get("batch")
    if not isinstance(discovery_descriptor, dict) or not isinstance(
        batch_descriptor, dict
    ):
        raise ValueError("discovery manifest descriptors are invalid")
    object_relative = discovery_descriptor.get("object_path")
    batch_relative = batch_descriptor.get("index_path")
    if not isinstance(object_relative, str) or not isinstance(batch_relative, str):
        raise ValueError("manifest object paths are invalid")
    object_path = root / object_relative
    batch_path = root / batch_relative
    if _sha256(object_path) != discovery_descriptor.get("object_sha256"):
        raise ValueError("discovery object hash differs")
    if _sha256(batch_path) != batch_descriptor.get("index_sha256"):
        raise ValueError("batch index hash differs")
    batch = _read_json(batch_path)
    if (
        batch.get("game_count") != 153
        or batch.get("final_holdout_access") != "CLOSED"
        or batch.get("publication_gate") != "PASS"
    ):
        raise ValueError("batch is not the exact sealed 153-game development set")

    totals = _stage_totals(root, batch)
    frame = pq.read_table(object_path).to_pandas()
    state = build_report_state(
        frame,
        game_count=153,
        actual_market_observations=totals["actual_market_observations"],
        reaction_paths=totals["reaction_paths"],
        factor_observations=totals["factor_observations"],
        source_sha256=str(discovery_descriptor["object_sha256"]),
    )
    state.update(
        {
            "batch_sha256": batch["batch_sha256"],
            "batch_index_sha256": batch_descriptor["index_sha256"],
            "discovery_manifest_sha256": _sha256(manifest_path),
        }
    )

    output = args.output.resolve()
    html_payload = render_exact153_html(frame, state).encode("utf-8")
    markdown_payload = render_exact153_markdown(frame, state).encode("utf-8")
    state_payload = _canonical_bytes(state)
    _atomic_write(output / "NFL_X13_153_GAME_REPORT.html", html_payload)
    _atomic_write(output / "NFL_X13_153_GAME_REPORT_ZH.md", markdown_payload)
    _atomic_write(output / "NFL_X13_153_GAME_STATE.json", state_payload)
    report_manifest = {
        "schema": "nfl_x13_exact153_report_manifest_v1",
        "experiment_id": "X-13",
        "cohort": "development",
        "game_count": 153,
        "holdout_access": "CLOSED",
        "source": {
            "batch_sha256": batch["batch_sha256"],
            "discovery_object_sha256": discovery_descriptor["object_sha256"],
            "discovery_manifest_sha256": _sha256(manifest_path),
        },
        "outputs": {
            "html": {
                "path": "NFL_X13_153_GAME_REPORT.html",
                "sha256": "sha256:" + hashlib.sha256(html_payload).hexdigest(),
                "byte_length": len(html_payload),
            },
            "markdown": {
                "path": "NFL_X13_153_GAME_REPORT_ZH.md",
                "sha256": "sha256:"
                + hashlib.sha256(markdown_payload).hexdigest(),
                "byte_length": len(markdown_payload),
            },
            "state": {
                "path": "NFL_X13_153_GAME_STATE.json",
                "sha256": "sha256:" + hashlib.sha256(state_payload).hexdigest(),
                "byte_length": len(state_payload),
            },
        },
        "claim_boundary": state["claim_boundary"],
    }
    _atomic_write(output / "manifest.json", _canonical_bytes(report_manifest))
    print(json.dumps(report_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
