"""Deterministic metadata-only selector for the X-13 historical holdout."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from prediction_market.sports.nfl_x13_holdout_evidence import (
    HoldoutEvidenceError,
    verify_holdout_evidence_index,
)

HOLDOUT_SEED = 20260725
HOLDOUT_SIZE = 20
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GAME_ID_RE = re.compile(r"2025_[0-9]{2}_[A-Z]+_[A-Z]+\Z")
_MAX_EVIDENCE_INDEX_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class HistoricalHoldoutSelection:
    status: str
    seed: int
    eligible_game_count: int
    eligible_game_ids: tuple[str, ...]
    selected_game_ids: tuple[str, ...]
    evidence_index_sha256: str | None
    evidence_manifest_sha256s: tuple[str, ...]
    blockers: tuple[str, ...]
    selection_id: str


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _not_frozen(
    *,
    eligible: tuple[str, ...],
    evidence_index_sha256: str | None,
    evidence_sha256s: tuple[str, ...],
    blockers: set[str],
) -> HistoricalHoldoutSelection:
    material = {
        "status": "NOT_FROZEN_MISSING_COVERAGE_EVIDENCE",
        "seed": HOLDOUT_SEED,
        "eligible_game_ids": list(eligible),
        "selected_game_ids": [],
        "evidence_index_sha256": evidence_index_sha256,
        "evidence_manifest_sha256s": list(evidence_sha256s),
        "blockers": sorted(blockers),
    }
    return HistoricalHoldoutSelection(
        status="NOT_FROZEN_MISSING_COVERAGE_EVIDENCE",
        seed=HOLDOUT_SEED,
        eligible_game_count=len(eligible),
        eligible_game_ids=eligible,
        selected_game_ids=(),
        evidence_index_sha256=evidence_index_sha256,
        evidence_manifest_sha256s=evidence_sha256s,
        blockers=tuple(sorted(blockers)),
        selection_id=_canonical_sha256(material),
    )


def select_historical_holdout(
    *,
    evidence_index_path: str | Path,
    store_root: str | Path,
    program_root: str | Path,
    discovery_game_ids: tuple[str, ...],
) -> HistoricalHoldoutSelection:
    """Freeze 20 games only after byte-exact evidence-index verification."""

    if (
        type(discovery_game_ids) is not tuple
        or len(set(discovery_game_ids)) != len(discovery_game_ids)
        or any(
            type(game_id) is not str or _GAME_ID_RE.fullmatch(game_id) is None
            for game_id in discovery_game_ids
        )
    ):
        raise ValueError("discovery_game_ids must be unique canonical 2025 IDs")
    index_path = Path(evidence_index_path)
    try:
        if index_path.is_symlink() or not index_path.is_file():
            raise OSError("evidence index is not a regular file")
        size = index_path.stat().st_size
        if size <= 0 or size > _MAX_EVIDENCE_INDEX_BYTES:
            raise OSError("evidence index byte length is invalid")
        with index_path.open("rb") as source:
            index_bytes = source.read(size + 1)
        if len(index_bytes) != size:
            raise OSError("evidence index changed while reading")
        verified = verify_holdout_evidence_index(
            index_bytes,
            store_root=store_root,
            program_root=program_root,
        )
    except (OSError, HoldoutEvidenceError, ValueError):
        return _not_frozen(
            eligible=(),
            evidence_index_sha256=None,
            evidence_sha256s=(),
            blockers={"MISSING_OR_INVALID_EVIDENCE_INDEX"},
        )
    index_sha256 = verified.get("index_sha256")
    games = verified.get("games")
    if (
        type(index_sha256) is not str
        or _SHA256_RE.fullmatch(index_sha256) is None
        or type(games) is not list
        or verified.get("game_count") != len(games)
    ):
        return _not_frozen(
            eligible=(),
            evidence_index_sha256=None,
            evidence_sha256s=(),
            blockers={"MISSING_OR_INVALID_EVIDENCE_INDEX"},
        )

    discovery = set(discovery_game_ids)
    eligible_ids: list[str] = []
    evidence_digests: set[str] = set()
    for game in games:
        if type(game) is not dict:
            continue
        game_id = game.get("game_id")
        if (
            type(game_id) is not str
            or _GAME_ID_RE.fullmatch(game_id) is None
            or game_id in discovery
        ):
            continue
        references = (
            game.get("schedule_evidence"),
            game.get("polymarket"),
            game.get("kalshi"),
        )
        manifests = [
            reference.get("manifest_sha256")
            for reference in references
            if type(reference) is dict
        ]
        if (
            len(manifests) != 3
            or any(
                type(digest) is not str
                or _SHA256_RE.fullmatch(digest) is None
                for digest in manifests
            )
        ):
            continue
        eligible_ids.append(game_id)
        evidence_digests.update(manifests)
    eligible = tuple(sorted(set(eligible_ids)))
    evidence_sha256s = tuple(sorted(evidence_digests))
    if len(eligible) < HOLDOUT_SIZE:
        return _not_frozen(
            eligible=eligible,
            evidence_index_sha256=index_sha256,
            evidence_sha256s=evidence_sha256s,
            blockers={"FEWER_THAN_20_ELIGIBLE_GAMES"},
        )
    selected = tuple(
        sorted(
            eligible,
            key=lambda game_id: (
                hashlib.sha256(
                    f"{HOLDOUT_SEED}|{game_id}".encode()
                ).hexdigest(),
                game_id,
            ),
        )[:HOLDOUT_SIZE]
    )
    material = {
        "status": "FROZEN",
        "seed": HOLDOUT_SEED,
        "selected_game_ids": list(selected),
        "evidence_index_sha256": index_sha256,
        "evidence_manifest_sha256s": list(evidence_sha256s),
    }
    return HistoricalHoldoutSelection(
        status="FROZEN",
        seed=HOLDOUT_SEED,
        eligible_game_count=len(eligible),
        eligible_game_ids=eligible,
        selected_game_ids=selected,
        evidence_index_sha256=index_sha256,
        evidence_manifest_sha256s=evidence_sha256s,
        blockers=(),
        selection_id=_canonical_sha256(material),
    )


def write_historical_holdout_selection(
    selection: HistoricalHoldoutSelection,
    *,
    output_root: str | Path,
) -> Path:
    """Publish one immutable, content-addressed holdout decision artifact."""

    if not isinstance(selection, HistoricalHoldoutSelection):
        raise TypeError("selection must be HistoricalHoldoutSelection")
    payload_value = {
        "artifact_type": "nfl_x13_historical_holdout_selection",
        "selection_id": selection.selection_id,
        "status": selection.status,
        "seed": selection.seed,
        "eligible_game_count": selection.eligible_game_count,
        "eligible_game_ids": list(selection.eligible_game_ids),
        "selected_game_ids": list(selection.selected_game_ids),
        "evidence_index_sha256": selection.evidence_index_sha256,
        "evidence_manifest_sha256s": list(
            selection.evidence_manifest_sha256s
        ),
        "blockers": list(selection.blockers),
        "claim_boundary": {
            "reaction_used": False,
            "trade_price_used": False,
            "volume_used": False,
            "event_result_used": False,
        },
    }
    payload = json.dumps(
        payload_value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode() + b"\n"
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / selection.selection_id.replace(":", "-")
    name = (
        f"{selection.selection_id.removeprefix('sha256:')}"
        ".holdout-selection.json"
    )
    target_path = target / name
    checksum_payload = (
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
    ).encode("ascii")
    if target.exists():
        if (
            target.is_symlink()
            or not target_path.is_file()
            or target_path.read_bytes() != payload
            or (target / "checksums.sha256").read_bytes() != checksum_payload
        ):
            raise ValueError("immutable holdout selection artifact conflicts")
        return target_path
    stage = Path(tempfile.mkdtemp(prefix=".x13-holdout-", dir=root))
    try:
        for path, content in (
            (stage / name, payload),
            (stage / "checksums.sha256", checksum_payload),
        ):
            with path.open("xb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        os.rename(stage, target)
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if stage.exists():
            stage.rmdir()
    return target_path


__all__ = [
    "HOLDOUT_SEED",
    "HOLDOUT_SIZE",
    "HistoricalHoldoutSelection",
    "select_historical_holdout",
    "write_historical_holdout_selection",
]
