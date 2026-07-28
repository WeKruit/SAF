"""Single-game-first materialization for the NFL X-13 Factor Lab.

The orchestration in this module has one deliberate ordering constraint:
every single-game bundle is built and verified before any cross-game
calculation is allowed to run.  The production loader below reads only the
already-frozen X-13 batch and verified Factor Lab inputs.  It never fetches an
upstream source and it never opens the frozen holdout reaction set.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prediction_market.immutable_object_store import ImmutableObjectStore
from prediction_market.sports.nfl_factor_lab import (
    VerifiedX13InputsV1,
    verify_and_load_x13_inputs,
)
from prediction_market.sports.nfl_factor_lab_analysis import (
    run_factor_discovery,
)
from prediction_market.sports.nfl_factor_lab_artifacts import (
    verify_factor_discovery_bundle,
    verify_sports_feature_view,
)
from prediction_market.sports.nfl_factor_lab_bundles import (
    PublishedCrossGameBundleV2,
    PublishedSingleGameBundleV2,
    _CROSS_STAGE_GRAINS,
    _SINGLE_STAGE_GRAINS,
    build_cross_game_bundle,
    build_single_game_bundle,
    mirror_bundle,
    verify_cross_game_bundle,
    verify_single_game_bundle,
)
from prediction_market.sports.nfl_factor_lab_cleaning import (
    build_sports_row_disposition_ledger,
)
from prediction_market.sports.nfl_factor_lab_factor_observations import (
    build_factor_observations_v2,
)
from prediction_market.sports.nfl_factor_lab_run_index import (
    VerifiedFactorLabRunIndexV2,
    build_factor_lab_run_index,
    factor_lab_materializer_code_lock_sha256,
)
from prediction_market.sports.nfl_x13_association import (
    build_game_time_intervals_v1,
    build_market_time_intervals_v1,
)
from prediction_market.sports.nfl_x13_batch import (
    _validate_game_payload,
    canonical_payload_hash,
)
from prediction_market.sports.nfl_x13_market import (
    MarketObservation,
)
from prediction_market.sports.nfl_x13_market_cleaning import (
    X13MarketRecleanGameV2,
    reclean_x13_verified_market_game_v2,
    verify_x13_market_capture_v2,
)
from prediction_market.sports.nfl_x13_replay import (
    CanonicalGameEventV1,
    CumulativeStatLedgerEntryV1,
    FinalizedEpisodeV1,
    GameStateSnapshotV1,
    X13GameLedgerV1,
)
from prediction_market.sports.nfl_x13_spec import (
    MarketContractV1,
    X13_BATCH_SPEC,
    X13_DELAY_SCENARIOS_SECONDS,
    X13_GAME_BINDINGS,
    X13_GAME_IDS,
    X13_HORIZONS_SECONDS,
    canonical_sha256,
)


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FROZEN_BATCH_ID = (
    "sha256:3ddafb13fe58faa3caf01fa720a60f6c2efc4e87d9977b129b3c3824b43c1d85"
)
_FROZEN_MARKET_CAPTURE_SHA256 = (
    "sha256:d78eaf7d203df942959748b942bdf7215f5cd8ae83e4967774843fad1953d516"
)
_FROZEN_MARKET_CAPTURE_RECEIPT_SHA256 = (
    "sha256:f08c3a175cc6a91c6eeec92de2cac745977b6f90aa53df4e59054ba8d1e6a53d"
)
_MARKET_RECLEAN_STAGE_SEMANTICS = "RAW_CAPTURE_V2_RECLEAN_EQUIVALENT"
_AUTHORITY_SOURCE_RELATIVE_PATHS = (
    (
        "catalog_registry_current",
        Path("charter/catalog_registry.csv"),
        "current_catalog_registry_sha256",
    ),
    (
        "data_license_register_current",
        Path("registries/data_license_register.csv"),
        "current_data_license_registry_sha256",
    ),
    (
        "dataset_registry_current",
        Path("registries/dataset_registry.csv"),
        "current_dataset_registry_sha256",
    ),
    (
        "experiment_registry_current",
        Path("registries/experiment_registry.csv"),
        "current_experiment_registry_sha256",
    ),
)
_MARKET_RECLEAN_SOURCE_ARTIFACTS = (
    *(
        name
        for name, _relative, _audit_field
        in _AUTHORITY_SOURCE_RELATIVE_PATHS
    ),
    "x13_market_capture_pointer_v2",
    "x13_market_capture_receipt_v2",
    "x13_market_universe_v1",
)
_MARKET_UNIVERSE_RELATIVE_PATH = Path(
    "artifacts/market-observation/nfl/x13/universe/"
    "sha256-01526155f3436a752cd6765d5b6a88086ab71badac84e82518a526cc8a7098d2/"
    "x13-universe-plan-v1.json"
)
_MARKET_CAPTURE_POINTER_RELATIVE_PATH = Path(
    "var/raw/capture-pointers/"
    "d78eaf7d203df942959748b942bdf7215f5cd8ae83e4967774843fad1953d516.json"
)
_MARKET_CAPTURE_RECEIPT_RELATIVE_PATH = Path(
    "var/raw/capture-receipts/"
    "f08c3a175cc6a91c6eeec92de2cac745977b6f90aa53df4e59054ba8d1e6a53d.json"
)
_EXPECTED_V2_FEATURE_ROW_COUNT = 3_205
_PARTICIPATION_OBJECT_SHA256 = (
    "sha256:3b26b9487267f011fdbafbce133800ce964c963fdb8b27725abf65663cc46e0f"
)
_PARTICIPATION_MANIFEST_SHA256 = (
    "sha256:c5b2f2d56b3e04ce8ce6674e09a2571eef39bfefaeb16d2c06da50a332b1e291"
)
_V2_FEATURE_ROWS_SHA256 = (
    "sha256:5604a0913b441af319d84b3631283531380413cdc7840ee8df8446b1a36ffb09"
)
_V2_FEATURE_CODE_LOCK_SHA256 = (
    "sha256:c0f7d543bc6c621b0c40ee21703a124b2cc8daea8c1c18fb83353cb6f488c54c"
)
_EXPECTED_BASE_FACTOR_COUNT = 104
_EXPECTED_INTERACTION_FACTOR_COUNT = 1_368
_NFLVERSE_MANIFEST_SHA256_BY_SEASON = {
    2022: (
        "sha256:"
        "2cb979d9c9d4cb1f2b2a8503dfa1cf044bd0a1360fd5c2e9f67391a52f1986f6"
    ),
    2023: (
        "sha256:"
        "58d950dc79281bed7adbc5c580bd1a1f050636c30e1a149cb9165585dc01cf47"
    ),
    2024: (
        "sha256:"
        "1064c52aeb1d676809d6412f2816e65a83ce9e6a2862e8537cd11dc2eee29bdf"
    ),
    2025: (
        "sha256:"
        "e06985cfae03b5967d8970be8a9f7a82699d72b549595c044d1056b553c1d99d"
    ),
}

SingleGameFrameBuilder = Callable[
    [str, X13MarketRecleanGameV2],
    Mapping[str, pd.DataFrame],
]
CrossGameFrameBuilder = Callable[
    [tuple[PublishedSingleGameBundleV2, ...]], Mapping[str, pd.DataFrame]
]


class FactorLabMaterializationError(ValueError):
    """A materialization plan or stage violated the frozen X-13 contract."""


def _partition_registered_factor_definitions(
    definitions: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Separate executable base factors from unapproved interaction inventory."""

    if not isinstance(definitions, Sequence):
        raise FactorLabMaterializationError(
            "factor definitions must be a sequence"
        )
    normalized: list[dict[str, object]] = []
    factor_ids: list[str] = []
    for raw in definitions:
        if not isinstance(raw, Mapping):
            raise FactorLabMaterializationError(
                "factor definition must be a mapping"
            )
        value = dict(raw)
        factor_id = value.get("factor_id")
        if type(factor_id) is not str or not factor_id:
            raise FactorLabMaterializationError(
                "factor definition has no stable factor_id"
            )
        normalized.append(value)
        factor_ids.append(factor_id)
    if len(set(factor_ids)) != len(factor_ids):
        raise FactorLabMaterializationError(
            "factor definition inventory contains duplicate factor_id"
        )
    base = tuple(
        value
        for value in normalized
        if ".INTERACTION." not in str(value["factor_id"])
    )
    interactions = tuple(
        value
        for value in normalized
        if ".INTERACTION." in str(value["factor_id"])
    )
    if (
        len(base) != _EXPECTED_BASE_FACTOR_COUNT
        or len(interactions) != _EXPECTED_INTERACTION_FACTOR_COUNT
    ):
        raise FactorLabMaterializationError(
            "frozen registry must contain exactly 104 base and "
            "1368 interaction factor definitions"
        )
    return base, interactions


@dataclass(frozen=True, slots=True)
class FactorLabMaterializationPlanV2:
    project_root: Path
    single_game_output_root: Path
    cross_game_output_root: Path
    run_index_path: Path
    game_ids: tuple[str, ...]
    source_hashes: tuple[str, ...]
    source_artifacts: Mapping[str, Path]
    factor_registry_sha256: str
    code_lock_sha256: str
    batch_spec_sha256: str
    s3_bucket: str
    s3_prefix: str

    def __post_init__(self) -> None:
        if (
            not self.source_hashes
            or tuple(sorted(set(self.source_hashes))) != self.source_hashes
        ):
            raise FactorLabMaterializationError(
                "source_hashes must be sorted, unique, and nonempty"
            )
        for digest in (
            *self.source_hashes,
            self.factor_registry_sha256,
            self.code_lock_sha256,
            self.batch_spec_sha256,
        ):
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise FactorLabMaterializationError(
                    "materialization hashes must be canonical SHA-256 values"
                )
        if not self.source_artifacts:
            raise FactorLabMaterializationError("source_artifacts must not be empty")
        missing_market_sources = set(
            _MARKET_RECLEAN_SOURCE_ARTIFACTS
        ).difference(self.source_artifacts)
        if missing_market_sources:
            raise FactorLabMaterializationError(
                "market reclean source artifacts are missing: "
                + ", ".join(sorted(missing_market_sources))
            )
        for (
            source_name,
            relative,
            _audit_field,
        ) in _AUTHORITY_SOURCE_RELATIVE_PATHS:
            if Path(self.source_artifacts[source_name]).resolve() != (
                self.project_root / relative
            ).resolve():
                raise FactorLabMaterializationError(
                    f"authority source path differs: {source_name}"
                )
        if not self.s3_bucket or not self.s3_prefix:
            raise FactorLabMaterializationError(
                "S3 bucket and prefix must be explicit"
            )


@dataclass(frozen=True, slots=True)
class MaterializedFactorLabV2:
    single_game_bundles: tuple[PublishedSingleGameBundleV2, ...]
    cross_game_bundle: PublishedCrossGameBundleV2
    run_index: VerifiedFactorLabRunIndexV2


@dataclass(frozen=True, slots=True)
class ExistingX13MaterializationInputsV2:
    """Verified bindings to the current frozen discovery evidence."""

    verified_inputs: VerifiedX13InputsV1
    batch_root: Path
    batch_manifest_path: Path
    batch_manifest: Mapping[str, object]
    checksums: Mapping[str, str]
    game_paths: Mapping[str, Path]
    participation_manifest_path: Path
    participation_object_path: Path
    participation_manifest_sha256: str
    participation_manifest_file_sha256: str
    participation_object_sha256: str
    feature_view_manifest_path: Path
    feature_view_data_path: Path
    feature_view_rows_sha256: str
    feature_view_code_lock_sha256: str
    source_artifacts: Mapping[str, Path]
    source_hashes: tuple[str, ...]


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_length = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            byte_length += len(chunk)
    return "sha256:" + digest.hexdigest(), byte_length


def _canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _read_canonical_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FactorLabMaterializationError(
            f"canonical JSON is unreadable: {path}"
        ) from error
    if not isinstance(value, dict):
        raise FactorLabMaterializationError(
            f"canonical JSON root is not an object: {path}"
        )
    if raw != _canonical_json_bytes(value, newline=raw.endswith(b"\n")):
        raise FactorLabMaterializationError(
            f"JSON input is not canonical: {path}"
        )
    return value


def _checksum_ledger(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text("ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise FactorLabMaterializationError(
            "frozen batch checksum ledger is unreadable"
        ) from error
    values: dict[str, str] = {}
    for line in lines:
        if not line:
            continue
        if len(line) < 67 or line[64:66] != "  ":
            raise FactorLabMaterializationError(
                "frozen batch checksum ledger is malformed"
            )
        digest = line[:64]
        relative = line[66:]
        if (
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in values
        ):
            raise FactorLabMaterializationError(
                "frozen batch checksum ledger contains an unsafe entry"
            )
        values[relative] = "sha256:" + digest
    if not values:
        raise FactorLabMaterializationError(
            "frozen batch checksum ledger is empty"
        )
    return values


def _verify_file_from_ledger(
    root: Path,
    checksums: Mapping[str, str],
    relative: str,
) -> Path:
    expected = checksums.get(relative)
    path = root / relative
    if expected is None or not path.is_file() or path.is_symlink():
        raise FactorLabMaterializationError(
            f"frozen batch object is absent from its checksum ledger: {relative}"
        )
    observed, _length = _sha256_path(path)
    if observed != expected:
        raise FactorLabMaterializationError(
            f"frozen batch object hash differs: {relative}"
        )
    return path


def _verified_participation_source(
    data_checkout_root: Path,
) -> tuple[Path, Path, str, str, str]:
    """Resolve and byte-verify the frozen research-only participation object."""

    manifest_root = (
        data_checkout_root
        / "var/raw/manifests/source=nflverse"
        / "dataset=DS-NFLVERSE-PARTICIPATION"
    )
    candidates = tuple(
        sorted(manifest_root.glob("*/partition=season-2025/*.manifest.json"))
    )
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            continue
        payload = _read_canonical_json(path)
        if payload.get("manifest_sha256") == _PARTICIPATION_MANIFEST_SHA256:
            matches.append((path, payload))
    if len(matches) != 1:
        raise FactorLabMaterializationError(
            "frozen 2025 participation manifest is missing or ambiguous"
        )
    manifest_path, manifest = matches[0]
    digest_payload = dict(manifest)
    digest_payload.pop("manifest_sha256", None)
    observed_manifest_sha = "sha256:" + hashlib.sha256(
        _canonical_json_bytes(digest_payload, newline=False)
    ).hexdigest()
    if (
        observed_manifest_sha != _PARTICIPATION_MANIFEST_SHA256
        or manifest.get("dataset_id") != "DS-NFLVERSE-PARTICIPATION"
        or manifest.get("object_sha256") != _PARTICIPATION_OBJECT_SHA256
        or manifest.get("license_status") != "research_only"
        or manifest.get("upstream_partition") != "season-2025"
        or manifest.get("object_kind") != "byte_exact_original"
    ):
        raise FactorLabMaterializationError(
            "frozen participation manifest identity or research boundary differs"
        )
    native_path = manifest.get("native_object_path")
    if (
        not isinstance(native_path, str)
        or not native_path
        or Path(native_path).is_absolute()
        or ".." in Path(native_path).parts
    ):
        raise FactorLabMaterializationError(
            "participation manifest has an unsafe native object path"
        )
    raw_root = (data_checkout_root / "var/raw").resolve()
    object_path = (raw_root / native_path).resolve()
    try:
        object_path.relative_to(raw_root)
    except ValueError as error:
        raise FactorLabMaterializationError(
            "participation object escapes the immutable raw root"
        ) from error
    if object_path.is_symlink() or not object_path.is_file():
        raise FactorLabMaterializationError(
            "frozen participation object is not locally available"
        )
    observed_object_sha, observed_byte_length = _sha256_path(object_path)
    if (
        observed_object_sha != _PARTICIPATION_OBJECT_SHA256
        or observed_byte_length != manifest.get("byte_length")
    ):
        raise FactorLabMaterializationError(
            "frozen participation object hash or byte length differs"
        )
    parquet = pq.ParquetFile(object_path)
    required_columns = {
        "nflverse_game_id",
        "play_id",
        "offense_players",
        "defense_players",
        "offense_names",
        "defense_names",
    }
    if (
        parquet.metadata.num_rows != 45_184
        or not required_columns.issubset(parquet.schema_arrow.names)
    ):
        raise FactorLabMaterializationError(
            "frozen participation object schema or row count differs"
        )
    manifest_file_sha, _ = _sha256_path(manifest_path)
    return (
        manifest_path,
        object_path,
        _PARTICIPATION_MANIFEST_SHA256,
        manifest_file_sha,
        _PARTICIPATION_OBJECT_SHA256,
    )


def _verified_nflverse_manifest_paths(
    project_root: Path,
    verified: VerifiedX13InputsV1,
) -> dict[str, Path]:
    """Bind the four local PIT inputs to their approved static manifests."""

    paths_by_season: dict[int, Path] = {}
    for object_path in verified.nflverse_pbp_paths:
        partition = next(
            (
                part
                for part in object_path.parts
                if part.startswith("partition=season-")
            ),
            None,
        )
        if partition is None:
            raise FactorLabMaterializationError(
                "nflverse PBP object has no season partition"
            )
        try:
            season = int(partition.removeprefix("partition=season-"))
        except ValueError as error:
            raise FactorLabMaterializationError(
                "nflverse PBP season partition is invalid"
            ) from error
        if season in paths_by_season:
            raise FactorLabMaterializationError(
                "nflverse PBP season partition is duplicated"
            )
        paths_by_season[season] = object_path
    if set(paths_by_season) != set(_NFLVERSE_MANIFEST_SHA256_BY_SEASON):
        raise FactorLabMaterializationError(
            "nflverse PIT object coverage differs from 2022-2025"
        )

    result: dict[str, Path] = {}
    manifest_base = (
        project_root
        / "var/raw/manifests/source=nflverse"
        / "dataset=DS-NFLVERSE"
        / "version=github-release-58152862-20260212T102526Z"
    )
    for season, expected_manifest_sha in sorted(
        _NFLVERSE_MANIFEST_SHA256_BY_SEASON.items()
    ):
        manifest_path = (
            manifest_base
            / f"partition=season-{season}"
            / (
                expected_manifest_sha.removeprefix("sha256:")
                + ".manifest.json"
            )
        )
        manifest = _read_canonical_json(manifest_path)
        material = dict(manifest)
        material.pop("manifest_sha256", None)
        semantic_sha = "sha256:" + hashlib.sha256(
            _canonical_json_bytes(material, newline=False)
        ).hexdigest()
        object_path = paths_by_season[season]
        object_sha = "sha256:" + object_path.stem
        if (
            semantic_sha != expected_manifest_sha
            or manifest.get("manifest_sha256") != expected_manifest_sha
            or manifest.get("dataset_id") != "DS-NFLVERSE"
            or manifest.get("license_status") != "approved"
            or manifest.get("license_ref") != "O-010"
            or manifest.get("object_kind") != "byte_exact_original"
            or manifest.get("upstream_partition") != f"season-{season}"
            or manifest.get("object_sha256") != object_sha
            or object_sha not in verified.source_hashes
        ):
            raise FactorLabMaterializationError(
                f"nflverse season-{season} manifest identity differs"
            )
        result[f"nflverse_{season}_manifest"] = manifest_path
    return result


def load_verified_existing_x13_materialization_inputs(
    project_root: str | Path,
) -> ExistingX13MaterializationInputsV2:
    """Bind the frozen discovery inputs without fetching or reading holdout reaction."""

    root = Path(project_root).resolve()
    verified = verify_and_load_x13_inputs(root)
    if (
        verified.discovery_game_ids != X13_GAME_IDS
        or verified.holdout_reaction_accessed
    ):
        raise FactorLabMaterializationError(
            "verified X-13 inputs do not preserve the frozen discovery boundary"
        )
    batch_root = (
        root
        / "artifacts/market-observation/nfl/x13/batches"
        / _FROZEN_BATCH_ID.replace(":", "-")
    )
    manifest_path = batch_root / "batch_manifest.json"
    checksum_path = batch_root / "checksums.sha256"
    manifest = _read_canonical_json(manifest_path)
    checksums = _checksum_ledger(checksum_path)
    _verify_file_from_ledger(batch_root, checksums, "batch_manifest.json")
    if (
        manifest.get("schema") != "nfl_x13_batch_manifest_v1"
        or manifest.get("batch_id") != _FROZEN_BATCH_ID
        or manifest.get("plan_id") != X13_BATCH_SPEC.plan_id
        or manifest.get("publication_gate") != "PASS"
        or manifest.get("status") != "PRELIMINARY_SOURCE_TIME_ONLY"
        or tuple(manifest.get("game_ids", ())) != X13_GAME_IDS
        or manifest.get("game_count") != 20
    ):
        raise FactorLabMaterializationError(
            "batch manifest does not bind the frozen X-13 discovery set"
        )
    game_hashes = manifest.get("game_hashes")
    if not isinstance(game_hashes, Mapping) or set(game_hashes) != set(
        X13_GAME_IDS
    ):
        raise FactorLabMaterializationError(
            "batch manifest game hashes do not cover the frozen set"
        )
    game_paths: dict[str, Path] = {}
    physical_hashes: list[str] = []
    expected_game_entries = {
        f"games/{game_id}.json" for game_id in X13_GAME_IDS
    }
    observed_game_entries = {
        relative
        for relative in checksums
        if relative.startswith("games/") and relative.endswith(".json")
    }
    if observed_game_entries != expected_game_entries:
        raise FactorLabMaterializationError(
            "frozen checksum ledger game coverage differs"
        )
    for game_id in X13_GAME_IDS:
        relative = f"games/{game_id}.json"
        expected_physical_hash = checksums.get(relative)
        path = batch_root / relative
        try:
            path.resolve().relative_to(batch_root.resolve())
        except ValueError as error:
            raise FactorLabMaterializationError(
                f"frozen game path escapes batch root: {game_id}"
            ) from error
        if (
            expected_physical_hash is None
            or path.is_symlink()
            or not path.is_file()
        ):
            raise FactorLabMaterializationError(
                f"frozen game path is absent or unsafe: {game_id}"
            )
        game_paths[game_id] = path
        physical_hashes.append(expected_physical_hash)

    (
        participation_manifest_path,
        participation_object_path,
        participation_manifest_sha,
        participation_manifest_file_sha,
        participation_object_sha,
    ) = _verified_participation_source(verified.data_checkout_root)
    feature_manifest_path = (
        root
        / "artifacts/market-observation/nfl/x13/factor-lab"
        / "sports-feature-view"
        / _V2_FEATURE_ROWS_SHA256.replace(":", "-")
        / "manifest.json"
    )
    published_feature_view = verify_sports_feature_view(
        feature_manifest_path
    )
    if (
        published_feature_view.row_count != _EXPECTED_V2_FEATURE_ROW_COUNT
        or published_feature_view.rows_sha256 != _V2_FEATURE_ROWS_SHA256
        or published_feature_view.code_lock_sha256
        != _V2_FEATURE_CODE_LOCK_SHA256
        or published_feature_view.factor_registry_sha256
        != verified.factor_registry_sha256
    ):
        raise FactorLabMaterializationError(
            "published V2 Sports Feature View identity differs"
        )

    prior_feature_manifest_path = (
        root
        / "artifacts/market-observation/nfl/x13/factor-lab"
        / "sports-feature-view"
        / (
            "sha256-"
            "38d431567e9206662e4f92f2931c3c1df141c2d06972e6c420b2b5e86770a8f3"
        )
        / "manifest.json"
    )
    prior_discovery_manifest_path = (
        root
        / "artifacts/market-observation/nfl/x13/factor-lab"
        / "discovery"
        / (
            "sha256-"
            "33a11366f28823ea9e36274db82c82d2ad6976b489cf246d862b05fe9a8d1a55"
        )
        / "manifest.json"
    )
    prior_feature_view = verify_sports_feature_view(
        prior_feature_manifest_path
    )
    prior_discovery = verify_factor_discovery_bundle(
        prior_discovery_manifest_path
    )
    if (
        prior_feature_view.factor_registry_sha256
        != verified.factor_registry_sha256
        or prior_discovery.factor_registry_sha256
        != verified.factor_registry_sha256
        or prior_discovery.bundle_sha256
        != (
            "sha256:"
            "33a11366f28823ea9e36274db82c82d2ad6976b489cf246d862b05fe9a8d1a55"
        )
    ):
        raise FactorLabMaterializationError(
            "prior X-13 Factor Lab evidence identity differs"
        )

    source_artifacts: dict[str, Path] = {
        **{
            name: root / relative
            for name, relative, _audit_field
            in _AUTHORITY_SOURCE_RELATIVE_PATHS
        },
        "factor_registry_v1": verified.factor_registry_path,
        "method_catalog_v1": verified.method_catalog_path,
        "x13_market_capture_pointer_v2": (
            root / _MARKET_CAPTURE_POINTER_RELATIVE_PATH
        ),
        "x13_market_capture_receipt_v2": (
            root / _MARKET_CAPTURE_RECEIPT_RELATIVE_PATH
        ),
        "x13_market_universe_v1": root / _MARKET_UNIVERSE_RELATIVE_PATH,
        "x13_signal_manifest_v1": verified.signal_manifest_path,
        "x13_correlation_cache_manifest_v1": verified.cache_manifest_path,
        "x13_reaction_paths_v1": (
            verified.actual_observation_paths_path
        ),
        "x13_prior_sports_feature_view_manifest_v1": (
            prior_feature_view.manifest_path
        ),
        "x13_prior_discovery_manifest_v1": (
            prior_discovery.manifest_path
        ),
        "x13_batch_manifest_v1": manifest_path,
        "x13_sports_feature_view_v2_manifest": (
            published_feature_view.manifest_path
        ),
        **_verified_nflverse_manifest_paths(root, verified),
    }
    for label, path in source_artifacts.items():
        if not path.is_file():
            raise FactorLabMaterializationError(
                f"materialization source artifact is missing: {label}"
            )
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise FactorLabMaterializationError(
                f"run-index source artifact is outside project root: {label}"
            ) from error
    manifest_file_hash, _ = _sha256_path(manifest_path)
    checksum_file_hash, _ = _sha256_path(checksum_path)
    feature_manifest_file_hash, _ = _sha256_path(
        published_feature_view.manifest_path
    )
    feature_data_file_hash, _ = _sha256_path(
        published_feature_view.data_path
    )
    source_artifact_hashes = tuple(
        _sha256_path(path)[0] for path in source_artifacts.values()
    )
    source_hashes = tuple(
        sorted(
            {
                *verified.source_hashes,
                _FROZEN_BATCH_ID,
                _FROZEN_MARKET_CAPTURE_SHA256,
                _FROZEN_MARKET_CAPTURE_RECEIPT_SHA256,
                manifest_file_hash,
                checksum_file_hash,
                participation_manifest_sha,
                participation_manifest_file_sha,
                participation_object_sha,
                published_feature_view.rows_sha256,
                published_feature_view.code_lock_sha256,
                feature_manifest_file_hash,
                feature_data_file_hash,
                *source_artifact_hashes,
                *(str(value) for value in game_hashes.values()),
                *physical_hashes,
            }
        )
    )
    return ExistingX13MaterializationInputsV2(
        verified_inputs=verified,
        batch_root=batch_root,
        batch_manifest_path=manifest_path,
        batch_manifest=manifest,
        checksums=checksums,
        game_paths=game_paths,
        participation_manifest_path=participation_manifest_path,
        participation_object_path=participation_object_path,
        participation_manifest_sha256=participation_manifest_sha,
        participation_manifest_file_sha256=participation_manifest_file_sha,
        participation_object_sha256=participation_object_sha,
        feature_view_manifest_path=published_feature_view.manifest_path,
        feature_view_data_path=published_feature_view.data_path,
        feature_view_rows_sha256=published_feature_view.rows_sha256,
        feature_view_code_lock_sha256=published_feature_view.code_lock_sha256,
        source_artifacts=source_artifacts,
        source_hashes=source_hashes,
    )


def _definition_records(
    inputs: VerifiedX13InputsV1,
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for definition in inputs.factor_definitions:
        value = definition.to_dict()
        value.pop("schema_version")
        records.append(value)
    return tuple(records)


def _decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise FactorLabMaterializationError("decimal evidence cannot be boolean")
    try:
        result = Decimal(str(value))
    except Exception as error:
        raise FactorLabMaterializationError(
            f"decimal evidence is invalid: {value!r}"
        ) from error
    if not result.is_finite():
        raise FactorLabMaterializationError("decimal evidence is non-finite")
    return result


def _snapshot(value: object) -> GameStateSnapshotV1 | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise FactorLabMaterializationError("game-state snapshot is not an object")
    names = {item.name for item in fields(GameStateSnapshotV1)}
    if set(value) != names:
        raise FactorLabMaterializationError(
            "game-state snapshot fields differ from GameStateSnapshotV1"
        )
    return GameStateSnapshotV1(**dict(value))


def _typed_game_ledger(
    payload: Mapping[str, object],
) -> X13GameLedgerV1:
    events_value = payload.get("events")
    episodes_value = payload.get("episodes")
    stats_value = payload.get("stat_ledger")
    if not all(isinstance(value, list) for value in (events_value, episodes_value, stats_value)):
        raise FactorLabMaterializationError(
            "frozen game replay arrays are missing"
        )
    events: list[CanonicalGameEventV1] = []
    for raw in events_value:  # type: ignore[union-attr]
        if not isinstance(raw, Mapping):
            raise FactorLabMaterializationError("canonical event is not an object")
        value = dict(raw)
        value["pre_state"] = _snapshot(value.get("pre_state"))
        value["post_state"] = _snapshot(value.get("post_state"))
        value["offense_names"] = tuple(value.get("offense_names", ()))
        value["defense_names"] = tuple(value.get("defense_names", ()))
        expected = {item.name for item in fields(CanonicalGameEventV1)}
        if set(value) != expected or value["pre_state"] is None:
            raise FactorLabMaterializationError(
                "canonical event fields differ from CanonicalGameEventV1"
            )
        events.append(CanonicalGameEventV1(**value))
    episodes: list[FinalizedEpisodeV1] = []
    for raw in episodes_value:  # type: ignore[union-attr]
        if not isinstance(raw, Mapping):
            raise FactorLabMaterializationError("finalized episode is not an object")
        value = dict(raw)
        value["play_ids"] = tuple(value.get("play_ids", ()))
        value["pre_state"] = _snapshot(value.get("pre_state"))
        value["post_state"] = _snapshot(value.get("post_state"))
        expected = {item.name for item in fields(FinalizedEpisodeV1)}
        allowed_batch_evidence = {
            "contaminated",
            "source_time_end_utc",
            "source_time_start_utc",
        }
        if (
            set(value) - expected != allowed_batch_evidence
            or value["pre_state"] is None
        ):
            raise FactorLabMaterializationError(
                "finalized episode fields differ from FinalizedEpisodeV1"
            )
        episodes.append(
            FinalizedEpisodeV1(
                **{name: value[name] for name in expected}
            )
        )
    stat_ledger: list[CumulativeStatLedgerEntryV1] = []
    for raw in stats_value:  # type: ignore[union-attr]
        if not isinstance(raw, Mapping):
            raise FactorLabMaterializationError("stat ledger row is not an object")
        value = dict(raw)
        period_scores = value.get("period_scores")
        if not isinstance(period_scores, Mapping):
            raise FactorLabMaterializationError(
                "stat ledger period_scores is not an object"
            )
        value["period_scores"] = {
            int(period): dict(scores)
            for period, scores in period_scores.items()
            if isinstance(scores, Mapping)
        }
        expected = {item.name for item in fields(CumulativeStatLedgerEntryV1)}
        if set(value) != expected:
            raise FactorLabMaterializationError(
                "stat ledger fields differ from CumulativeStatLedgerEntryV1"
            )
        stat_ledger.append(CumulativeStatLedgerEntryV1(**value))
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping):
        raise FactorLabMaterializationError("frozen game lineage is missing")
    ledger_sha = lineage.get("game_state_ledger_sha256")
    if not isinstance(ledger_sha, str) or _SHA256_RE.fullmatch(ledger_sha) is None:
        raise FactorLabMaterializationError(
            "frozen game-state ledger hash is missing"
        )
    return X13GameLedgerV1(
        events=tuple(events),
        episodes=tuple(episodes),
        stat_ledger=tuple(stat_ledger),
        events_sha256=ledger_sha,
        artifact_sha256=ledger_sha,
    )


def _typed_contracts(
    payload: Mapping[str, object],
) -> tuple[MarketContractV1, ...]:
    values = payload.get("contracts")
    if not isinstance(values, list) or not values:
        raise FactorLabMaterializationError("frozen contract inventory is missing")
    contracts: list[MarketContractV1] = []
    expected = {item.name for item in fields(MarketContractV1)}
    for raw in values:
        if not isinstance(raw, Mapping):
            raise FactorLabMaterializationError("market contract is not an object")
        value = {name: raw.get(name) for name in expected}
        value["line"] = _decimal(value["line"])
        value["outcomes"] = tuple(value["outcomes"] or ())
        value["dependency_game_ids"] = tuple(
            value["dependency_game_ids"] or ()
        )
        contracts.append(MarketContractV1(**value))
    return tuple(contracts)


def _formal_market_inputs_v2(
    *,
    game_id: str,
    market_game: X13MarketRecleanGameV2,
    current_authority_snapshot_sha256: str,
    current_authority_component_sha256s: Mapping[str, str],
) -> tuple[
    dict[str, pd.DataFrame],
    tuple[MarketContractV1, ...],
    tuple[MarketObservation, ...],
    dict[str, str],
]:
    """Validate and adapt one already-recleaned game without reopening raw."""

    if (
        getattr(market_game, "status", None) != "PASS"
        or getattr(market_game, "game_id", None) != game_id
    ):
        raise FactorLabMaterializationError(
            "recleaned market game identity or status differs"
        )
    capture_sha256 = getattr(market_game, "capture_sha256", None)
    if (
        not isinstance(capture_sha256, str)
        or _SHA256_RE.fullmatch(capture_sha256) is None
    ):
        raise FactorLabMaterializationError(
            "formal market capture identity is invalid"
        )
    authority_paths = tuple(
        relative.as_posix()
        for _name, relative, _audit_field
        in _AUTHORITY_SOURCE_RELATIVE_PATHS
    )
    if (
        not isinstance(current_authority_component_sha256s, Mapping)
        or set(current_authority_component_sha256s) != set(authority_paths)
        or any(
            not isinstance(
                current_authority_component_sha256s[path],
                str,
            )
            or _SHA256_RE.fullmatch(
                current_authority_component_sha256s[path]
            )
            is None
            for path in authority_paths
        )
        or not isinstance(current_authority_snapshot_sha256, str)
        or _SHA256_RE.fullmatch(current_authority_snapshot_sha256) is None
    ):
        raise FactorLabMaterializationError(
            "formal current authority identity is invalid"
        )
    expected_authority_snapshot_sha256 = canonical_sha256(
        {
            "schema": "x13-current-dataset-authority-v2",
            "components": [
                {
                    "path": path,
                    "sha256": (
                        current_authority_component_sha256s[path]
                    ),
                }
                for path in authority_paths
            ],
        }
    )
    if (
        current_authority_snapshot_sha256
        != expected_authority_snapshot_sha256
    ):
        raise FactorLabMaterializationError(
            "formal current authority snapshot identity differs"
        )
    try:
        contract_records = market_game.contract_inventory_records()
        observation_records = market_game.normalized_observation_records()
        audit_records = market_game.market_cleaning_audit_records()
        typed_observations = tuple(market_game.normalized_observations)
    except (AttributeError, TypeError) as error:
        raise FactorLabMaterializationError(
            "recleaned market game does not expose the V2 record API"
        ) from error
    for label, values in (
        ("contract inventory", contract_records),
        ("normalized observations", observation_records),
        ("market cleaning audit", audit_records),
    ):
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(row, Mapping) for row in values)
        ):
            raise FactorLabMaterializationError(
                f"formal {label} records are missing"
            )
        if {row.get("game_id") for row in values} != {game_id}:
            raise FactorLabMaterializationError(
                f"formal {label} contains another game"
            )
    if (
        not typed_observations
        or any(
            not isinstance(value, MarketObservation)
            for value in typed_observations
        )
    ):
        raise FactorLabMaterializationError(
            "recleaned normalized observations are not typed"
        )

    typed_contracts = _typed_contracts(
        {"contracts": contract_records}
    )
    logical_market_by_contract: dict[str, str] = {}
    exploded_contracts: list[dict[str, object]] = []
    for raw in contract_records:
        contract_id = raw.get("contract_id")
        logical_market_id = raw.get("logical_market_id")
        outcomes = raw.get("outcomes")
        native_rule_hash = raw.get("rule_sha256")
        if (
            not isinstance(contract_id, str)
            or not contract_id
            or not isinstance(logical_market_id, str)
            or not logical_market_id
            or (
                contract_id in logical_market_by_contract
                and logical_market_by_contract[contract_id]
                != logical_market_id
            )
            or not isinstance(outcomes, list)
            or not outcomes
            or any(not isinstance(outcome, str) or not outcome for outcome in outcomes)
            or not isinstance(native_rule_hash, str)
            or _SHA256_RE.fullmatch(native_rule_hash) is None
        ):
            raise FactorLabMaterializationError(
                "formal contract inventory identity is invalid"
            )
        logical_market_by_contract[contract_id] = logical_market_id
        for outcome in outcomes:
            exploded_contracts.append(
                {
                    **dict(raw),
                    "outcome": outcome,
                    "native_rule_hash": native_rule_hash,
                }
            )
    if len(logical_market_by_contract) != len(contract_records):
        raise FactorLabMaterializationError(
            "formal contract inventory contains duplicate contract_id"
        )

    observations = pd.DataFrame(
        [dict(row) for row in observation_records]
    )
    market_audit = pd.DataFrame([dict(row) for row in audit_records])
    contracts = pd.DataFrame(exploded_contracts)
    observation_ids = tuple(
        observations["observation_id"].astype(str)
    )
    typed_observation_ids = tuple(
        observation.observation_id
        for observation in typed_observations
    )
    if (
        observations.duplicated(["game_id", "observation_id"]).any()
        or observation_ids != typed_observation_ids
        or not observations["stage_semantics"].eq(
            _MARKET_RECLEAN_STAGE_SEMANTICS
        ).all()
    ):
        raise FactorLabMaterializationError(
            "formal normalized market observation identity differs"
        )
    expected_contract_ids = set(logical_market_by_contract)
    if (
        not set(observations["contract_id"].astype(str)).issubset(
            expected_contract_ids
        )
        or not observations.apply(
            lambda row: logical_market_by_contract.get(
                str(row["contract_id"])
            )
            == row["logical_market_id"],
            axis=1,
        ).all()
    ):
        raise FactorLabMaterializationError(
            "formal normalized market observation contract binding differs"
        )

    required_audit_columns = {
        "actual_observation",
        "actual_trade",
        "capture_sha256",
        "captured_license_status",
        "condition_id",
        "current_authority_snapshot_sha256",
        "current_catalog_registry_sha256",
        "current_data_license_registry_sha256",
        "current_dataset_registry_sha256",
        "current_experiment_registry_sha256",
        "current_license_status",
        "data_gap",
        "dedupe_decision",
        "included",
        "observation_id",
        "raw_cursor_in",
        "raw_cursor_out",
        "raw_manifest_sha256",
        "raw_object_sha256",
        "raw_page_index",
        "raw_request_row_available",
        "raw_request_sha256",
        "raw_row_fingerprint",
        "raw_row_index",
        "raw_source_cursor",
        "raw_source_url",
        "raw_token_or_ticker_binding_available",
        "stage_semantics",
        "terminal_cursor",
        "terminal_manifest_sha256",
        "terminal_proof",
        "ticker",
        "token_id",
        "v2_cleaning_equivalent",
    }
    missing_audit_columns = required_audit_columns.difference(
        market_audit.columns
    )
    if missing_audit_columns:
        raise FactorLabMaterializationError(
            "formal market audit lineage columns are missing: "
            + ", ".join(sorted(missing_audit_columns))
        )
    kept_audit = market_audit[market_audit["included"].eq(True)].copy()
    if (
        set(market_audit["observation_id"].astype(str))
        != set(observation_ids)
        or kept_audit.duplicated(["observation_id"]).any()
        or set(kept_audit["observation_id"].astype(str))
        != set(observation_ids)
    ):
        raise FactorLabMaterializationError(
            "formal market audit does not cover exact normalized observations"
        )
    for flag in (
        "raw_request_row_available",
        "raw_token_or_ticker_binding_available",
        "v2_cleaning_equivalent",
    ):
        if not all(value is True for value in market_audit[flag].tolist()):
            raise FactorLabMaterializationError(
                f"formal market audit has a false {flag} flag"
            )
    expected_authority_audit_values = {
        "current_authority_snapshot_sha256": (
            current_authority_snapshot_sha256
        ),
        **{
            audit_field: current_authority_component_sha256s[
                relative.as_posix()
            ]
            for _source_name, relative, audit_field
            in _AUTHORITY_SOURCE_RELATIVE_PATHS
        },
    }
    if (
        not market_audit["stage_semantics"].eq(
            _MARKET_RECLEAN_STAGE_SEMANTICS
        ).all()
        or not market_audit["data_gap"].isna().all()
        or not market_audit["capture_sha256"].eq(capture_sha256).all()
        or any(
            not market_audit[field].eq(digest).all()
            for field, digest in expected_authority_audit_values.items()
        )
    ):
        raise FactorLabMaterializationError(
            "formal market audit current authority or current dataset "
            "registry/V2 semantics differ"
        )
    for field in (
        "raw_manifest_sha256",
        "raw_object_sha256",
        "raw_request_sha256",
        "raw_row_fingerprint",
        "terminal_manifest_sha256",
    ):
        if not market_audit[field].map(
            lambda value: (
                isinstance(value, str)
                and _SHA256_RE.fullmatch(value) is not None
            )
        ).all():
            raise FactorLabMaterializationError(
                f"formal market audit {field} lineage is invalid"
            )
    for field in (
        "captured_license_status",
        "current_license_status",
        "raw_source_cursor",
        "raw_source_url",
        "terminal_proof",
    ):
        if not market_audit[field].map(
            lambda value: isinstance(value, str) and bool(value)
        ).all():
            raise FactorLabMaterializationError(
                f"formal market audit {field} lineage is invalid"
            )
    for field in ("raw_page_index", "raw_row_index"):
        if not market_audit[field].map(
            lambda value: type(value) is int and value >= 0
        ).all():
            raise FactorLabMaterializationError(
                f"formal market audit {field} lineage is invalid"
            )
    native_binding = market_audit.apply(
        lambda row: any(
            isinstance(row[field], str) and bool(row[field])
            for field in ("token_id", "ticker")
        ),
        axis=1,
    )
    if not native_binding.all():
        raise FactorLabMaterializationError(
            "formal market audit lacks token-or-ticker lineage"
        )

    semantics = kept_audit[
        ["observation_id", "actual_observation", "actual_trade"]
    ].merge(
        observations[["observation_id", "provenance", "kind"]],
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    expected_actual = semantics["provenance"].eq("observed")
    expected_trade = expected_actual & semantics["kind"].eq("trade")
    if (
        not semantics["actual_observation"].eq(expected_actual).all()
        or not semantics["actual_trade"].eq(expected_trade).all()
    ):
        raise FactorLabMaterializationError(
            "formal market actual/derived semantics differ"
        )
    return (
        {
            "contract_inventory": contracts,
            "market_observations": observations,
            "market_cleaning_audit": market_audit,
        },
        typed_contracts,
        typed_observations,
        logical_market_by_contract,
    )


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(child) for child in value.tolist()]
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _arrow_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep formal columns while encoding genuinely nested cells canonically."""

    result = frame.copy(deep=False)
    for column in result.columns:
        if result[column].dtype != object:
            continue
        values = result[column].tolist()
        if not any(
            isinstance(value, (Mapping, list, tuple, np.ndarray))
            for value in values
            if value is not None
        ):
            continue
        result[column] = [
            (
                json.dumps(
                    _json_value(value),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                if isinstance(value, (Mapping, list, tuple, np.ndarray))
                else value
            )
            for value in values
        ]
    return result


def _stable_play_id(value: object) -> str:
    if isinstance(value, bool) or value is None:
        raise FactorLabMaterializationError("play_id is missing")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    if isinstance(value, str) and value and value == value.strip():
        return value
    raise FactorLabMaterializationError(f"play_id is not stable: {value!r}")


def _optional_source_text(value: object) -> str | None:
    if value is None or (
        isinstance(value, (float, np.floating)) and math.isnan(float(value))
    ):
        return None
    if not isinstance(value, str):
        raise FactorLabMaterializationError(
            "player identity field is not text"
        )
    value = value.strip()
    return value or None


def _semicolon_values(value: object) -> tuple[str, ...]:
    text = _optional_source_text(value)
    if text is None:
        return ()
    values = tuple(child.strip() for child in text.split(";") if child.strip())
    if len(values) != len(set(values)):
        raise FactorLabMaterializationError(
            "participation field contains a duplicate identity"
        )
    return values


def _sequence_values(value: object, *, field: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if (
        isinstance(value, (float, np.floating))
        and math.isnan(float(value))
    ):
        return ()
    raise FactorLabMaterializationError(f"{field} is not a sequence")


def _participation_identity(
    row: Mapping[str, object] | None,
    *,
    side: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    if row is None:
        return (), (), "DATA_GAP_NO_PARTICIPATION_ROW"
    ids = _semicolon_values(row.get(f"{side}_players"))
    names = _semicolon_values(row.get(f"{side}_names"))
    if not ids or not names or len(ids) != len(names):
        return (), (), "DATA_GAP_PARTICIPATION_ID_NAME_ALIGNMENT"
    pairs = tuple(sorted(zip(ids, names, strict=True), key=lambda item: item[0]))
    return (
        tuple(player_id for player_id, _name in pairs),
        tuple(player_name for _id, player_name in pairs),
        "PARTICIPATION_IDS_BOUND",
    )


def _explicit_player_roles(
    source: Mapping[str, object],
    legacy_roles: object,
) -> tuple[dict[str, object], str]:
    """Use only native PBP role IDs; never reverse-map a legacy name."""

    from prediction_market.sports.nfl_factor_lab_features import _ROLE_FIELDS

    legacy = legacy_roles if isinstance(legacy_roles, Mapping) else {}
    roles: dict[str, object] = {}
    for role, id_field, name_field in _ROLE_FIELDS:
        player_id = _optional_source_text(source.get(id_field))
        player_name = _optional_source_text(source.get(name_field))
        if player_id is None:
            continue
        legacy_name = legacy.get(role)
        if (
            legacy_name is not None
            and player_name is not None
            and legacy_name != player_name
        ):
            raise FactorLabMaterializationError(
                f"explicit {role} player name conflicts with legacy replay"
            )
        roles[role] = {
            "player_id": player_id,
            "player_name": player_name,
        }
    return (
        dict(sorted(roles.items())),
        (
            "EXPLICIT_PBP_ROLE_IDS_BOUND"
            if roles
            else "DATA_GAP_NO_EXPLICIT_ROLE_ID"
        ),
    )


def _canonical_events_v2(
    *,
    game_id: str,
    legacy_events: Sequence[Mapping[str, object]],
    selected_pbp: pd.DataFrame,
    participation: pd.DataFrame,
    participation_source_sha256: str,
) -> pd.DataFrame:
    """Enrich every immutable legacy event with direct, audited player IDs."""

    pbp_by_play = {
        _stable_play_id(row["play_id"]): row
        for row in selected_pbp.to_dict("records")
    }
    participation_by_play = {
        _stable_play_id(row["play_id"]): row
        for row in participation.to_dict("records")
    }
    if len(pbp_by_play) != len(selected_pbp) or len(participation_by_play) != len(
        participation
    ):
        raise FactorLabMaterializationError(
            "PBP or participation identity is not unique by play"
        )
    rows: list[dict[str, object]] = []
    for legacy_event in legacy_events:
        row = dict(legacy_event)
        if row.get("game_id") != game_id:
            raise FactorLabMaterializationError(
                "legacy canonical event contains another game"
            )
        play_id = _stable_play_id(row.get("play_id"))
        source = pbp_by_play.get(play_id)
        if source is None:
            raise FactorLabMaterializationError(
                "canonical event cannot be traced to the selected raw PBP"
            )
        participant = participation_by_play.get(play_id)
        offense_ids, offense_names, offense_status = _participation_identity(
            participant,
            side="offense",
        )
        defense_ids, defense_names, defense_status = _participation_identity(
            participant,
            side="defense",
        )
        legacy_offense_names = tuple(
            str(value)
            for value in _sequence_values(
                row.get("offense_names"),
                field="legacy offense_names",
            )
        )
        legacy_defense_names = tuple(
            str(value)
            for value in _sequence_values(
                row.get("defense_names"),
                field="legacy defense_names",
            )
        )
        if offense_ids and set(offense_names) != set(legacy_offense_names):
            raise FactorLabMaterializationError(
                "participation offense names conflict with canonical replay"
            )
        if defense_ids and set(defense_names) != set(legacy_defense_names):
            raise FactorLabMaterializationError(
                "participation defense names conflict with canonical replay"
            )
        player_roles_v2, role_status = _explicit_player_roles(
            source,
            row.get("player_roles"),
        )
        participant_statuses = {offense_status, defense_status}
        if participant_statuses == {"PARTICIPATION_IDS_BOUND"}:
            participation_status = "PARTICIPATION_IDS_BOUND"
        elif "DATA_GAP_PARTICIPATION_ID_NAME_ALIGNMENT" in participant_statuses:
            participation_status = (
                "DATA_GAP_PARTICIPATION_ID_NAME_ALIGNMENT"
            )
            offense_ids = ()
            defense_ids = ()
            offense_names = ()
            defense_names = ()
        else:
            participation_status = "DATA_GAP_NO_PARTICIPATION_ROW"
        row.update(
            {
                "offense_player_ids": offense_ids,
                "defense_player_ids": defense_ids,
                "offense_player_names": (
                    offense_names if offense_names else legacy_offense_names
                ),
                "defense_player_names": (
                    defense_names if defense_names else legacy_defense_names
                ),
                "player_roles_v2": player_roles_v2,
                "player_identity_status": participation_status,
                "role_identity_status": role_status,
                "participation_license_status": "research_only",
                "participation_semantics": (
                    "post-season play participation; no exact substitution "
                    "or event-availability timestamp"
                ),
                "legacy_canonical_event_sha256": canonical_sha256(row),
                "participation_source_sha256": participation_source_sha256,
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if (
        len(result) != len(legacy_events)
        or result.duplicated(["game_id", "play_id"]).any()
    ):
        raise FactorLabMaterializationError(
            "V2 canonical event enrichment changed the canonical grain"
        )
    return result


def _pit_features_v2(
    *,
    feature_frame: pd.DataFrame,
    selected_pbp: pd.DataFrame,
    participation: pd.DataFrame,
    participation_source_sha256: str,
) -> pd.DataFrame:
    """Attach verified participation IDs without changing PIT feature rows."""

    game_ids = set(feature_frame["game_id"].astype(str))
    if len(game_ids) != 1:
        raise FactorLabMaterializationError(
            "PIT feature enrichment requires exactly one game"
        )
    game_id = next(iter(game_ids))
    pbp_by_play = {
        _stable_play_id(row["play_id"]): row
        for row in selected_pbp.to_dict("records")
    }
    participation_by_play = {
        _stable_play_id(row["play_id"]): row
        for row in participation.to_dict("records")
    }
    output = feature_frame.copy()
    offense_ids_values: list[tuple[str, ...]] = []
    defense_ids_values: list[tuple[str, ...]] = []
    offense_names_values: list[tuple[str, ...]] = []
    defense_names_values: list[tuple[str, ...]] = []
    roles_values: list[dict[str, object]] = []
    identity_statuses: list[str] = []
    role_statuses: list[str] = []
    for row in output.to_dict("records"):
        if row.get("game_id") != game_id:
            raise FactorLabMaterializationError(
                "PIT feature row contains another game"
            )
        play_id = _stable_play_id(row.get("play_id"))
        source = pbp_by_play.get(play_id)
        if source is None:
            raise FactorLabMaterializationError(
                "PIT feature cannot be traced to the selected raw PBP"
            )
        participant = participation_by_play.get(play_id)
        offense_ids, offense_names, offense_status = _participation_identity(
            participant,
            side="offense",
        )
        defense_ids, defense_names, defense_status = _participation_identity(
            participant,
            side="defense",
        )
        existing_offense_names = tuple(
            str(value)
            for value in _sequence_values(
                row.get("offense_player_names"),
                field="PIT offense_player_names",
            )
        )
        existing_defense_names = tuple(
            str(value)
            for value in _sequence_values(
                row.get("defense_player_names"),
                field="PIT defense_player_names",
            )
        )
        if offense_ids and set(offense_names) != set(existing_offense_names):
            raise FactorLabMaterializationError(
                "PIT offense names conflict with participation source"
            )
        if defense_ids and set(defense_names) != set(existing_defense_names):
            raise FactorLabMaterializationError(
                "PIT defense names conflict with participation source"
            )
        statuses = {offense_status, defense_status}
        if statuses == {"PARTICIPATION_IDS_BOUND"}:
            identity_status = "PARTICIPATION_IDS_BOUND"
        elif "DATA_GAP_PARTICIPATION_ID_NAME_ALIGNMENT" in statuses:
            identity_status = "DATA_GAP_PARTICIPATION_ID_NAME_ALIGNMENT"
            offense_ids = ()
            defense_ids = ()
        else:
            identity_status = "DATA_GAP_NO_PARTICIPATION_ROW"
        roles, role_status = _explicit_player_roles(
            source,
            {
                child.get("role"): child.get("player_name")
                for child in _sequence_values(
                    row.get("player_roles"),
                    field="PIT player_roles",
                )
                if isinstance(child, Mapping)
            },
        )
        offense_ids_values.append(tuple(sorted(offense_ids)))
        defense_ids_values.append(tuple(sorted(defense_ids)))
        offense_names_values.append(tuple(sorted(existing_offense_names)))
        defense_names_values.append(tuple(sorted(existing_defense_names)))
        roles_values.append(roles)
        identity_statuses.append(identity_status)
        role_statuses.append(role_status)
    output["offense_player_ids"] = offense_ids_values
    output["defense_player_ids"] = defense_ids_values
    output["offense_player_names"] = offense_names_values
    output["defense_player_names"] = defense_names_values
    output["player_roles_v2"] = roles_values
    output["player_identity_status"] = identity_statuses
    output["role_identity_status"] = role_statuses
    output["participation_source_sha256"] = participation_source_sha256
    output["participation_license_status"] = "research_only"
    if len(output) != len(feature_frame) or output.duplicated(
        ["game_id", "play_id"]
    ).any():
        raise FactorLabMaterializationError(
            "participation enrichment changed the PIT feature grain"
        )
    return output


def _interval_overlap_counts(frame: pd.DataFrame) -> np.ndarray:
    start = (
        pd.to_datetime(frame["interval_start_utc"], utc=True)
        .astype("int64")
        .to_numpy()
        // 1_000
    )
    end = (
        pd.to_datetime(frame["interval_end_utc"], utc=True)
        .astype("int64")
        .to_numpy()
        // 1_000
    )
    inclusive = frame["end_inclusive"].astype(bool).to_numpy()
    sorted_starts = np.sort(start)
    sorted_end_keys = np.sort(end * 2 + inclusive.astype(np.int64))
    started = np.searchsorted(
        sorted_starts,
        end,
        side="right",
    )
    exclusive = ~inclusive
    if exclusive.any():
        started[exclusive] = np.searchsorted(
            sorted_starts,
            end[exclusive],
            side="left",
        )
    ended = np.searchsorted(
        sorted_end_keys,
        start * 2,
        side="right",
    )
    counts = started - ended - 1
    if (counts < 0).any():
        raise FactorLabMaterializationError(
            "source timeline overlap count became negative"
        )
    return counts


def _build_source_timeline_frames(
    *,
    game_id: str,
    game_intervals: pd.DataFrame,
    market_intervals: pd.DataFrame,
    canonical_events: pd.DataFrame,
    finalized_episodes: pd.DataFrame,
    market_observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized equivalent of the persisted G/M timeline formal semantics."""

    episode_first_play = finalized_episodes[
        ["episode_id", "play_ids", "episode_type"]
    ].copy()
    episode_first_play["source_play_id"] = episode_first_play["play_ids"].map(
        lambda value: (
            value[0]
            if isinstance(value, (list, tuple)) and value
            else None
        )
    )
    event_context = canonical_events[
        [
            "play_id",
            "clock_seconds_remaining",
            "game_seconds_remaining",
            "home_score",
            "away_score",
            "possession_team",
            "offense_team",
            "defense_team",
            "quarter",
            "description",
        ]
    ].copy()
    game_context = episode_first_play.merge(
        event_context,
        left_on="source_play_id",
        right_on="play_id",
        how="left",
        validate="many_to_one",
    )
    game_base = game_intervals.merge(
        game_context,
        on=["episode_id", "source_play_id"],
        how="left",
        validate="many_to_one",
    )
    game_base["layer"] = "G"
    game_base["identity"] = game_base["episode_id"]
    game_base["kind"] = game_base["episode_type"]
    game_base["venue"] = None
    game_base["logical_market_id"] = None
    game_base["outcome"] = None
    game_base["actual_price"] = np.nan
    game_base["observation_kind"] = None

    market_context = market_observations[
        [
            "observation_id",
            "logical_market_id",
            "outcome",
            "price",
            "kind",
        ]
    ].copy()
    market_base = market_intervals.merge(
        market_context,
        on=["observation_id", "logical_market_id", "outcome"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_native"),
    )
    market_base["layer"] = "M"
    market_base["identity"] = market_base["observation_id"]
    market_base["kind"] = market_base["observation_kind"]
    for column in (
        "clock_seconds_remaining",
        "game_seconds_remaining",
        "home_score",
        "away_score",
        "possession_team",
        "offense_team",
        "defense_team",
        "quarter",
        "description",
        "episode_type",
        "source_play_id",
    ):
        market_base[column] = None
    market_base["actual_price"] = market_base["price"]

    columns = [
        "game_id",
        "layer",
        "identity",
        "interval_id",
        "interval_start_utc",
        "interval_end_utc",
        "end_inclusive",
        "delay_seconds",
        "kind",
        "venue",
        "provenance",
        "logical_market_id",
        "outcome",
        "actual_price",
        "observation_kind",
        "source_play_id",
        "episode_type",
        "quarter",
        "clock_seconds_remaining",
        "game_seconds_remaining",
        "home_score",
        "away_score",
        "possession_team",
        "offense_team",
        "defense_team",
        "description",
    ]
    timelines: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for delay in X13_DELAY_SCENARIOS_SECONDS:
        left = game_base[game_base["delay_seconds"].eq(delay)].copy()
        right = market_base.copy()
        right["delay_seconds"] = delay
        combined = pd.concat(
            [left[columns], right[columns]],
            ignore_index=True,
        ).sort_values(
            [
                "interval_start_utc",
                "interval_end_utc",
                "layer",
                "identity",
            ],
            kind="mergesort",
        ).reset_index(drop=True)
        overlap_count = _interval_overlap_counts(combined)
        combined["overlap_count"] = overlap_count
        combined["order_ambiguous"] = overlap_count > 0
        combined["forward_filled"] = False
        combined["asserted_order"] = None
        combined["timeline_id"] = (
            "timeline:v1:"
            + str(delay)
            + ":"
            + combined["layer"].astype(str)
            + ":"
            + combined["interval_id"].astype(str)
        )
        timelines.append(combined)
        ambiguous = combined[combined["order_ambiguous"]].copy()
        if not ambiguous.empty:
            same_start_count = ambiguous.groupby(
                "interval_start_utc", sort=False
            )["timeline_id"].transform("size")
            audit = ambiguous[
                [
                    "timeline_id",
                    "game_id",
                    "delay_seconds",
                    "layer",
                    "identity",
                    "overlap_count",
                ]
            ].copy()
            audit["reason_codes"] = np.where(
                same_start_count.gt(1),
                '["same_source_second","overlapping_source_intervals"]',
                '["overlapping_source_intervals"]',
            )
            audit["order_ambiguous"] = True
            audit["asserted_order"] = None
            audit["timeline_audit_id"] = (
                "timeline-audit:v1:" + audit["timeline_id"].astype(str)
            )
            audits.append(audit)
    timeline = pd.concat(timelines, ignore_index=True)
    if timeline.duplicated(["game_id", "timeline_id", "delay_seconds"]).any():
        raise FactorLabMaterializationError(
            "source timeline contains a duplicate formal grain"
        )
    timeline_audit = (
        pd.concat(audits, ignore_index=True)
        if audits
        else pd.DataFrame(
            columns=[
                "timeline_audit_id",
                "timeline_id",
                "game_id",
                "delay_seconds",
                "layer",
                "identity",
                "reason_codes",
                "overlap_count",
                "order_ambiguous",
                "asserted_order",
            ]
        )
    )
    return timeline, timeline_audit


def _association_attrition(
    frame: pd.DataFrame,
    *,
    logical_market_by_contract: Mapping[str, str],
    market_observations: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["logical_market_id"] = result["contract_id"].map(
        logical_market_by_contract
    )
    if result["logical_market_id"].isna().any():
        raise FactorLabMaterializationError(
            "association contract cannot be bound to a logical market"
        )
    result["association_id"] = (
        "association:v1:"
        + result["episode_id"].astype(str)
        + ":"
        + result["contract_id"].astype(str)
        + ":"
        + result["outcome"].astype(str)
        + ":"
        + result["horizon_seconds"].astype(str)
    )
    if result.duplicated(["game_id", "association_id"]).any():
        raise FactorLabMaterializationError(
            "association projection is not unique at its frozen delay=0 grain"
        )
    market_columns = {
        "contract_id",
        "kind",
        "outcome",
        "price",
        "primary_path_eligible",
        "provenance",
        "venue",
    }
    missing_market_columns = market_columns.difference(
        market_observations.columns
    )
    if missing_market_columns:
        raise FactorLabMaterializationError(
            "recleaned association evidence is missing columns: "
            + ", ".join(sorted(missing_market_columns))
        )
    actual_trade_series = set(
        market_observations[
            market_observations["kind"].eq("trade")
            & market_observations["provenance"].eq("observed")
            & market_observations["primary_path_eligible"].eq(True)
            & market_observations["price"].notna()
        ][["contract_id", "outcome", "venue"]].itertuples(
            index=False,
            name=None,
        )
    )
    result["recleaned_actual_trade_series_available"] = [
        (contract_id, outcome, venue) in actual_trade_series
        for contract_id, outcome, venue in result[
            ["contract_id", "outcome", "venue"]
        ].itertuples(index=False, name=None)
    ]
    conditions = [
        ~result["actual_observation"].eq(True),
        (
            result["actual_observation"].eq(True)
            & ~result["recleaned_actual_trade_series_available"]
        ),
        ~result["validity_status"].eq("OBSERVED"),
        result["contaminated"].eq(True),
        result["order_ambiguous"].eq(True),
        pd.to_numeric(
            result["staleness_seconds"], errors="coerce"
        ).gt(60),
        ~result["orientation_audited"].eq(True),
        ~result["analysis_eligible"].eq(True),
        ~result["signal_candidate_eligible"].eq(True),
    ]
    reasons = [
        "no_actual_trade",
        "no_recleaned_actual_trade_series",
        "validity_not_observed",
        "contaminated",
        "order_ambiguous",
        "stale_over_60s",
        "invalid_orientation",
        "analysis_ineligible",
        "signal_candidate_ineligible",
    ]
    result["exclusion_reason"] = np.select(
        conditions,
        reasons,
        default="",
    )
    result["strict_analysis_eligible"] = result[
        "exclusion_reason"
    ].eq("")
    result["attrition_status"] = np.where(
        result["strict_analysis_eligible"],
        "STRICT_ANALYSIS_ELIGIBLE",
        "EXCLUDED",
    )
    result["delay_seconds"] = 0
    return result


def _reaction_paths(
    frame: pd.DataFrame,
    *,
    logical_market_by_contract: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    identity = [
        "game_id",
        "episode_id",
        "contract_id",
        "outcome",
        "venue",
        "family",
        "path_complete",
        "path_factor_value",
    ]
    for horizon in X13_HORIZONS_SECONDS:
        child = frame[
            [
                *identity,
                f"change_{horizon}s",
                f"observed_{horizon}s",
                f"validity_{horizon}s",
            ]
        ].copy()
        child = child.rename(
            columns={
                f"change_{horizon}s": "signed_price_change",
                f"observed_{horizon}s": "actual_observation",
                f"validity_{horizon}s": "validity_status",
            }
        )
        child["horizon_seconds"] = horizon
        child["logical_market_id"] = child["contract_id"].map(
            logical_market_by_contract
        )
        rows.append(child)
    result = pd.concat(rows, ignore_index=True)
    if result["logical_market_id"].isna().any():
        raise FactorLabMaterializationError(
            "reaction path contract cannot be bound to a logical market"
        )
    return result


def _utc_text_from_ns(value: int) -> str:
    return (
        pd.Timestamp(value, unit="ns", tz="UTC")
        .isoformat()
        .replace("+00:00", "Z")
    )


def _seconds_from_ns(value: int) -> Decimal:
    return Decimal(value) / Decimal(1_000_000_000)


def _unique_signed_maximum_excursion(
    prices: Sequence[Decimal],
    *,
    pre_price: Decimal | None,
) -> Decimal | None:
    if pre_price is None or not prices:
        return None
    deltas = tuple(price - pre_price for price in prices)
    magnitude = max(abs(delta) for delta in deltas)
    candidates = {delta for delta in deltas if abs(delta) == magnitude}
    return next(iter(candidates)) if len(candidates) == 1 else None


def _build_reaction_paths_v2(
    *,
    actual_projection: pd.DataFrame,
    market_observations: pd.DataFrame,
    market_audit: pd.DataFrame,
    logical_market_by_contract: Mapping[str, str],
    market_source_hashes: tuple[str, ...],
) -> pd.DataFrame:
    """Recompute every path metric from recleaned primary-path actual trades."""

    candidate_columns = {
        "actual_observation",
        "analysis_eligible",
        "contaminated",
        "contract_id",
        "contract_measure",
        "episode_id",
        "family",
        "game_id",
        "horizon_seconds",
        "orientation_audited",
        "order_ambiguous",
        "outcome",
        "semantic_orientation_id",
        "signal_candidate_eligible",
        "signal_focal_team",
        "source_time_end_utc",
        "source_time_start_utc",
        "venue",
    }
    missing_candidate_columns = candidate_columns.difference(
        actual_projection.columns
    )
    if missing_candidate_columns:
        raise FactorLabMaterializationError(
            "actual reaction candidate columns are missing: "
            + ", ".join(sorted(missing_candidate_columns))
        )
    # This frozen true set is the approved 64,058-row candidate/audit
    # universe. No legacy metric, status, or eligibility value survives below.
    actual = actual_projection[
        actual_projection["actual_observation"].eq(True)
    ][sorted(candidate_columns)].copy()
    if actual.empty:
        raise FactorLabMaterializationError(
            "actual projection has no observed reaction path"
        )
    if not actual["horizon_seconds"].isin(X13_HORIZONS_SECONDS).all():
        raise FactorLabMaterializationError(
            "actual reaction candidate uses an unregistered horizon"
        )
    actual["logical_market_id"] = actual["contract_id"].map(
        logical_market_by_contract
    )
    if actual["logical_market_id"].isna().any():
        raise FactorLabMaterializationError(
            "actual reaction path has no logical market identity"
        )
    market_columns = {
        "contract_id",
        "kind",
        "observation_id",
        "outcome",
        "price",
        "primary_path_eligible",
        "provenance",
        "size",
        "source_end",
        "source_start",
        "venue",
    }
    missing_market_columns = market_columns.difference(
        market_observations.columns
    )
    if missing_market_columns:
        raise FactorLabMaterializationError(
            "recleaned reaction evidence is missing columns: "
            + ", ".join(sorted(missing_market_columns))
        )
    trades = market_observations[
        market_observations["kind"].eq("trade")
        & market_observations["provenance"].eq("observed")
        & market_observations["primary_path_eligible"].eq(True)
        & market_observations["price"].notna()
    ].copy()
    if trades.empty:
        raise FactorLabMaterializationError(
            "verified game projection contains no primary-path actual trade"
        )
    trades["source_start_ns"] = pd.to_datetime(
        trades["source_start"], utc=True, errors="raise"
    ).dt.as_unit("ns").astype("int64")
    trades["source_end_ns"] = pd.to_datetime(
        trades["source_end"], utc=True, errors="raise"
    ).dt.as_unit("ns").astype("int64")
    trades["price_decimal"] = trades["price"].map(_decimal)
    trades["size_decimal"] = trades["size"].map(_decimal)
    if (
        trades["observation_id"].duplicated().any()
        or trades["price_decimal"].map(
            lambda value: (
                isinstance(value, Decimal)
                and value.is_finite()
                and Decimal("0") <= value <= Decimal("1")
            )
        ).eq(False).any()
        or trades["size_decimal"].map(
            lambda value: (
                isinstance(value, Decimal)
                and value.is_finite()
                and value >= 0
            )
        ).eq(False).any()
        or trades["source_end_ns"].lt(trades["source_start_ns"]).any()
    ):
        raise FactorLabMaterializationError(
            "recleaned actual trade identity, numeric value, or interval is invalid"
        )

    lineage_fields = (
        "raw_manifest_sha256",
        "raw_object_sha256",
        "raw_request_sha256",
        "raw_row_fingerprint",
    )
    required_audit_columns = {
        "included",
        "observation_id",
        *lineage_fields,
    }
    missing_audit_columns = required_audit_columns.difference(
        market_audit.columns
    )
    if missing_audit_columns:
        raise FactorLabMaterializationError(
            "reaction metric audit lineage columns are missing: "
            + ", ".join(sorted(missing_audit_columns))
        )
    included_audit = market_audit[market_audit["included"].eq(True)].copy()
    if included_audit.duplicated(["observation_id"]).any():
        raise FactorLabMaterializationError(
            "reaction metric audit observation lineage is not unique"
        )
    lineage_by_observation = {
        str(row["observation_id"]): {
            field: row[field] for field in lineage_fields
        }
        for row in included_audit.to_dict("records")
    }
    selected_observation_ids = set(
        trades["observation_id"].astype(str)
    )
    if (
        not selected_observation_ids.issubset(lineage_by_observation)
        or any(
            not isinstance(value, str)
            or _SHA256_RE.fullmatch(value) is None
            for observation_id in selected_observation_ids
            for value in lineage_by_observation[observation_id].values()
        )
    ):
        raise FactorLabMaterializationError(
            "recleaned actual trades lack canonical direct audit lineage"
        )

    trade_indexes: dict[tuple[str, str, str], dict[str, object]] = {}
    for key, frame in trades.groupby(
        ["contract_id", "outcome", "venue"],
        sort=False,
        dropna=False,
    ):
        ordered = frame.sort_values(
            ["source_start_ns", "observation_id"],
            kind="mergesort",
        )
        starts = ordered["source_start_ns"].to_numpy(dtype=np.int64)
        ends = ordered["source_end_ns"].to_numpy(dtype=np.int64)
        end_order = np.argsort(ends, kind="stable")
        starts_by_end = starts[end_order]
        indexes_by_start: dict[int, list[int]] = {}
        for position, start in enumerate(starts):
            indexes_by_start.setdefault(int(start), []).append(position)
        trade_indexes[tuple(str(value) for value in key)] = {
            "starts": starts,
            "ends": ends,
            "prices": tuple(ordered["price_decimal"]),
            "sizes": tuple(ordered["size_decimal"]),
            "observation_ids": tuple(
                ordered["observation_id"].astype(str)
            ),
            "sorted_ends": ends[end_order],
            "latest_start_by_end_prefix": np.maximum.accumulate(
                starts_by_end
            ),
            "indexes_by_start": indexes_by_start,
        }

    event_start_ns = pd.to_datetime(
        actual["source_time_start_utc"], utc=True, errors="raise"
    ).dt.as_unit("ns").astype("int64").to_numpy(dtype=np.int64)
    event_end_ns = pd.to_datetime(
        actual["source_time_end_utc"], utc=True, errors="raise"
    ).dt.as_unit("ns").astype("int64").to_numpy(dtype=np.int64)
    if (
        (event_end_ns < event_start_ns).any()
        or actual.duplicated(
            [
                "game_id",
                "episode_id",
                "contract_id",
                "outcome",
                "venue",
                "horizon_seconds",
            ]
        ).any()
    ):
        raise FactorLabMaterializationError(
            "actual reaction candidate window or grain is invalid"
        )

    rebuilt_rows: list[dict[str, object]] = []
    for index, record in enumerate(actual.to_dict("records")):
        key = (
            str(record["contract_id"]),
            str(record["outcome"]),
            str(record["venue"]),
        )
        series = trade_indexes.get(key)
        event_start = int(event_start_ns[index])
        event_end = int(event_end_ns[index])
        horizon = int(record["horizon_seconds"])
        deadline = event_end + horizon * 1_000_000_000
        pre_index: int | None = None
        pre_ambiguous = False
        post_indexes: tuple[int, ...] = ()
        overlap_indexes: tuple[int, ...] = ()
        post_same_start_ambiguous = False
        if series is None:
            starts = np.asarray([], dtype=np.int64)
            ends = np.asarray([], dtype=np.int64)
            prices: tuple[Decimal, ...] = ()
            sizes: tuple[Decimal, ...] = ()
            observation_ids: tuple[str, ...] = ()
        else:
            starts = series["starts"]  # type: ignore[assignment]
            ends = series["ends"]  # type: ignore[assignment]
            prices = series["prices"]  # type: ignore[assignment]
            sizes = series["sizes"]  # type: ignore[assignment]
            observation_ids = series["observation_ids"]  # type: ignore[assignment]
            sorted_ends = series["sorted_ends"]  # type: ignore[assignment]
            prefix = series["latest_start_by_end_prefix"]  # type: ignore[assignment]
            indexes_by_start = series["indexes_by_start"]  # type: ignore[assignment]
            pre_stop = int(
                np.searchsorted(
                    sorted_ends,
                    event_start,
                    side="right",
                )
            )
            if pre_stop:
                latest_start = int(prefix[pre_stop - 1])
                candidates = tuple(
                    position
                    for position in indexes_by_start[latest_start]
                    if int(ends[position]) <= event_start
                )
                if len(candidates) == 1:
                    pre_index = candidates[0]
                else:
                    pre_ambiguous = True
            post_start = int(
                np.searchsorted(starts, event_end, side="right")
            )
            post_stop = int(
                np.searchsorted(starts, deadline, side="right")
            )
            post_indexes = tuple(range(post_start, post_stop))
            overlap_stop = int(
                np.searchsorted(starts, event_end, side="right")
            )
            overlap_indexes = tuple(
                position
                for position in range(overlap_stop)
                if int(ends[position]) > event_start
                and int(starts[position]) <= event_end
            )
            if len(post_indexes) > 1:
                post_starts = starts[post_start:post_stop]
                post_same_start_ambiguous = bool(
                    (np.diff(post_starts) == 0).any()
                )

        pre_price = (
            prices[pre_index] if pre_index is not None else None
        )
        post_prices = tuple(prices[position] for position in post_indexes)
        post_sizes = tuple(sizes[position] for position in post_indexes)
        volume = sum(post_sizes, Decimal("0"))
        horizon_vwap = (
            None
            if not post_indexes or volume == 0
            else sum(
                (
                    prices[position] * sizes[position]
                    for position in post_indexes
                ),
                Decimal("0"),
            )
            / volume
        )
        first_post_index: int | None = None
        last_post_index: int | None = None
        if post_indexes:
            first_start = int(starts[post_indexes[0]])
            first_candidates = tuple(
                position
                for position in post_indexes
                if int(starts[position]) == first_start
            )
            if len(first_candidates) == 1:
                first_post_index = first_candidates[0]
            last_start = int(starts[post_indexes[-1]])
            last_candidates = tuple(
                position
                for position in post_indexes
                if int(starts[position]) == last_start
            )
            if len(last_candidates) == 1:
                last_post_index = last_candidates[0]
        first_post_price = (
            prices[first_post_index]
            if first_post_index is not None
            else None
        )
        last_post_price = (
            prices[last_post_index]
            if last_post_index is not None
            else None
        )
        signed_price_change = (
            None
            if pre_price is None or last_post_price is None
            else last_post_price - pre_price
        )
        maximum_excursion = _unique_signed_maximum_excursion(
            post_prices,
            pre_price=pre_price,
        )
        staleness_seconds = (
            None
            if not post_indexes
            else _seconds_from_ns(
                deadline - int(starts[post_indexes[-1]])
            )
        )
        pre_trade_age_seconds = (
            None
            if pre_index is None
            else _seconds_from_ns(
                event_start - int(ends[pre_index])
            )
        )
        pre_trade_fresh = (
            pre_trade_age_seconds is not None
            and Decimal("0") <= pre_trade_age_seconds <= Decimal("60")
        )
        detected_ambiguity = bool(
            pre_ambiguous
            or post_same_start_ambiguous
            or overlap_indexes
        )
        order_ambiguous = (
            record["order_ambiguous"] is True or detected_ambiguity
        )
        contaminated = record["contaminated"] is True
        actual_observation = bool(
            pre_price is not None
            and first_post_price is not None
            and last_post_price is not None
            and horizon_vwap is not None
            and maximum_excursion is not None
            and post_indexes
            and not order_ambiguous
            and not contaminated
        )
        if contaminated:
            validity_status = "CONTAMINATED"
            gap_reason = "CONTAMINATED"
        elif order_ambiguous:
            validity_status = "ORDER_AMBIGUOUS"
            gap_reason = "ORDER_AMBIGUOUS_ACTUAL_TRADE_WINDOW"
        elif series is None:
            validity_status = "MISSING_OBSERVATION"
            gap_reason = "DATA_GAP_NO_MATCHING_ACTUAL_TRADE_SERIES"
        elif pre_price is None:
            validity_status = "MISSING_OBSERVATION"
            gap_reason = "DATA_GAP_NO_UNIQUE_PRE_EVENT_ACTUAL_TRADE"
        elif not post_indexes:
            validity_status = "MISSING_OBSERVATION"
            gap_reason = "DATA_GAP_NO_POST_EVENT_ACTUAL_TRADE"
        elif (
            first_post_price is None
            or last_post_price is None
            or horizon_vwap is None
            or maximum_excursion is None
        ):
            validity_status = "ORDER_AMBIGUOUS"
            gap_reason = "DATA_GAP_ACTUAL_TRADE_METRICS_UNDEFINED"
        else:
            validity_status = "OBSERVED"
            gap_reason = "RECOMPUTED_EXACT_ACTUAL_TRADE_WINDOW"

        candidate_age_valid = (
            pre_trade_age_seconds is not None
            and Decimal("0") <= pre_trade_age_seconds <= Decimal("60")
        )
        gate_conditions = (
            actual_observation,
            record["signal_candidate_eligible"] is True,
            record["outcome"] == record["signal_focal_team"],
            record["orientation_audited"] is True,
            record["contract_measure"] == "winner",
            record["family"] == "moneyline",
            pre_trade_fresh,
            candidate_age_valid,
        )
        analysis_eligible = all(gate_conditions)
        if analysis_eligible:
            analysis_exclusion_reason = "ELIGIBLE"
        elif validity_status != "OBSERVED":
            analysis_exclusion_reason = validity_status
        elif not post_indexes:
            analysis_exclusion_reason = "NO_ACTUAL_TRADE"
        elif record["signal_candidate_eligible"] is not True:
            analysis_exclusion_reason = "SIGNAL_CANDIDATE_INELIGIBLE"
        elif record["outcome"] != record["signal_focal_team"]:
            analysis_exclusion_reason = "NON_FOCAL_OUTCOME"
        elif record["orientation_audited"] is not True:
            analysis_exclusion_reason = "UNAUDITED_ORIENTATION"
        elif record["contract_measure"] != "winner":
            analysis_exclusion_reason = "NON_WINNER_MEASURE"
        elif record["family"] != "moneyline":
            analysis_exclusion_reason = "NON_MONEYLINE_FAMILY"
        elif not pre_trade_fresh:
            analysis_exclusion_reason = "STALE_PRE_TRADE"
        else:
            analysis_exclusion_reason = "INVALID_PRE_TRADE_AGE"

        metric_indexes = (
            *((pre_index,) if pre_index is not None else ()),
            *post_indexes,
        )
        metric_observation_ids = tuple(
            observation_ids[position] for position in metric_indexes
        )
        overlap_observation_ids = tuple(
            observation_ids[position] for position in overlap_indexes
        )
        lineage_hashes = {
            field: tuple(
                sorted(
                    {
                        lineage_by_observation[observation_id][field]
                        for observation_id in metric_observation_ids
                    }
                )
            )
            for field in lineage_fields
        }
        overlap_lineage_hashes = {
            field: tuple(
                sorted(
                    {
                        lineage_by_observation[observation_id][field]
                        for observation_id in overlap_observation_ids
                    }
                )
            )
            for field in lineage_fields
        }
        direct_hashes = {
            digest
            for values in (
                *lineage_hashes.values(),
                *overlap_lineage_hashes.values(),
            )
            for digest in values
        }
        net_change_60s = (
            signed_price_change if horizon == 60 else None
        )
        first_delta = (
            None
            if pre_price is None or first_post_price is None
            else first_post_price - pre_price
        )
        overshoot_candidate = bool(
            horizon == 60
            and actual_observation
            and maximum_excursion is not None
            and net_change_60s is not None
            and abs(maximum_excursion) > abs(net_change_60s)
        )
        reversal_candidate = bool(
            horizon == 60
            and actual_observation
            and first_delta is not None
            and net_change_60s is not None
            and first_delta * net_change_60s < 0
        )
        rebuilt_rows.append(
            {
                **record,
                "actual_observation": actual_observation,
                "analysis_eligible": analysis_eligible,
                "analysis_exclusion_reason": analysis_exclusion_reason,
                "contaminated": contaminated,
                "event_source_time_utc": record[
                    "source_time_start_utc"
                ],
                "first_post_trade_time_utc": (
                    _utc_text_from_ns(int(starts[first_post_index]))
                    if first_post_index is not None
                    else None
                ),
                "first_post_event_trade": first_post_price,
                "first_post_observation_id": (
                    observation_ids[first_post_index]
                    if first_post_index is not None
                    else None
                ),
                "forward_filled": False,
                "horizon_deadline_utc": _utc_text_from_ns(deadline),
                "horizon_last_trade": last_post_price,
                "horizon_last_trade_observation_id": (
                    observation_ids[last_post_index]
                    if last_post_index is not None
                    else None
                ),
                "horizon_trade_observation_ids": tuple(
                    observation_ids[position]
                    for position in post_indexes
                ),
                "horizon_vwap": horizon_vwap,
                "last_actual_trade": last_post_price,
                "market_source_hashes": tuple(
                    sorted(
                        {
                            *market_source_hashes,
                            *direct_hashes,
                        }
                    )
                ),
                "maximum_excursion": maximum_excursion,
                "metric_observation_ids": metric_observation_ids,
                "metric_observation_ids_sha256": canonical_sha256(
                    {
                        "schema": (
                            "nfl-x13-recomputed-reaction-metric-"
                            "observations-v2"
                        ),
                        "observation_ids": list(
                            metric_observation_ids
                        ),
                    }
                ),
                **{
                    f"metric_{field}s": values
                    for field, values in lineage_hashes.items()
                },
                "net_change_60s": net_change_60s,
                "order_ambiguous": order_ambiguous,
                "overlapping_actual_trade_observation_ids": (
                    overlap_observation_ids
                ),
                **{
                    f"overlap_{field}s": values
                    for field, values in overlap_lineage_hashes.items()
                },
                "overshoot_candidate": overshoot_candidate,
                "pre_event_actual_trade": pre_price,
                "pre_event_observation_id": (
                    observation_ids[pre_index]
                    if pre_index is not None
                    else None
                ),
                "pre_trade_age_seconds": pre_trade_age_seconds,
                "pre_trade_age_status": (
                    "RECONSTRUCTED_FROM_V2_ACTUAL_TRADES"
                    if pre_index is not None
                    else "MISSING_PRE_EVENT_ACTUAL_TRADE"
                ),
                "pre_trade_fresh_le_60s": pre_trade_fresh,
                "reaction_calculation_semantics": (
                    "V2_RECLEANED_PRIMARY_ACTUAL_TRADE_CUMULATIVE_WINDOW"
                ),
                "reaction_path_id": (
                    "reaction-path:v2:"
                    + str(record["game_id"])
                    + ":"
                    + str(record["episode_id"])
                    + ":"
                    + str(record["contract_id"])
                    + ":"
                    + str(record["outcome"])
                    + ":"
                    + str(record["venue"])
                    + ":"
                    + str(horizon)
                ),
                "reaction_time_verification_status": gap_reason,
                "reversal_candidate": reversal_candidate,
                "signed_price_change": signed_price_change,
                "staleness_seconds": staleness_seconds,
                "trade_count": len(post_indexes),
                "validity_status": validity_status,
                "volume": volume,
                "vwap": horizon_vwap,
            }
        )

    actual = pd.DataFrame(rebuilt_rows)
    pivot = actual.pivot_table(
        index=[
            "game_id",
            "episode_id",
            "contract_id",
            "outcome",
            "venue",
        ],
        columns="horizon_seconds",
        values="signed_price_change",
        aggfunc="first",
    )
    continuation = (
        (pivot[60] - pivot[10]).rename("continuation_10_to_60").reset_index()
        if 10 in pivot.columns and 60 in pivot.columns
        else pd.DataFrame(
            columns=[
                "game_id",
                "episode_id",
                "contract_id",
                "outcome",
                "venue",
                "continuation_10_to_60",
            ]
        )
    )
    actual = actual.merge(
        continuation,
        on=["game_id", "episode_id", "contract_id", "outcome", "venue"],
        how="left",
        validate="many_to_one",
    )
    direction = np.select(
        [
            pd.to_numeric(
                actual["signed_price_change"], errors="coerce"
            ).gt(0),
            pd.to_numeric(
                actual["signed_price_change"], errors="coerce"
            ).lt(0),
            pd.to_numeric(
                actual["signed_price_change"], errors="coerce"
            ).eq(0),
        ],
        ["UP", "DOWN", "FLAT"],
        default="MISSING",
    )
    actual["_recomputed_direction"] = direction
    consistency_keys = [
        "game_id",
        "episode_id",
        "semantic_orientation_id",
        "horizon_seconds",
    ]
    consistency: list[dict[str, object]] = []
    for key, frame in actual.groupby(
        consistency_keys,
        sort=False,
        dropna=False,
    ):
        venue_count = frame["venue"].nunique()
        eligible_directions = tuple(
            frame.loc[
                frame["actual_observation"].eq(True),
                "_recomputed_direction",
            ]
        )
        if venue_count == 1:
            status = "SINGLE_VENUE"
        elif len(eligible_directions) < 2:
            status = "INSUFFICIENT"
        elif len(set(eligible_directions)) == 1:
            status = "CONSISTENT"
        else:
            status = "DIVERGENT"
        consistency.append(
            {
                **dict(zip(consistency_keys, key, strict=True)),
                "two_venue_direction_consistency": status,
            }
        )
    actual = actual.merge(
        pd.DataFrame(consistency),
        on=consistency_keys,
        how="left",
        validate="many_to_one",
    ).drop(columns=["_recomputed_direction"])
    if actual.duplicated(
        [
            "game_id",
            "episode_id",
            "logical_market_id",
            "outcome",
            "venue",
            "horizon_seconds",
        ]
    ).any():
        raise FactorLabMaterializationError(
            "formal reaction paths contain duplicate grain"
        )
    return actual


def _episode_level_effects(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "factor_id",
        "market_family",
        "horizon_seconds",
        "game_id",
        "episode_id",
    ]
    required = {*keys, "signed_price_change"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise FactorLabMaterializationError(
            "game contribution input missing columns: "
            + ", ".join(missing)
        )
    effects = pd.to_numeric(
        observations["signed_price_change"],
        errors="coerce",
    )
    if effects.isna().any() or not np.isfinite(effects.to_numpy()).all():
        raise FactorLabMaterializationError(
            "game contribution requires finite signed price changes"
        )
    prepared = observations.loc[:, keys].copy()
    prepared["signed_price_change"] = effects.to_numpy(dtype=float)
    return (
        prepared.groupby(keys, sort=True, as_index=False)[
            "signed_price_change"
        ]
        .mean()
    )


def _build_game_contribution(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose the registered episode-level effect by contributing game."""

    episode_effects = _episode_level_effects(observations)
    keys = [
        "factor_id",
        "market_family",
        "horizon_seconds",
        "game_id",
    ]
    contribution = (
        episode_effects.groupby(keys, sort=True)["signed_price_change"]
        .agg(effect_sum="sum", point_estimate="mean", observation_count="size")
        .reset_index()
    )
    contribution = contribution.rename(
        columns={"observation_count": "episode_count"}
    )
    contribution["absolute_effect_sum"] = contribution["effect_sum"].abs()
    denominators = contribution.groupby(
        ["factor_id", "market_family", "horizon_seconds"],
        sort=False,
    )["absolute_effect_sum"].transform("sum")
    contribution["absolute_effect_denominator"] = denominators
    contribution["contribution"] = np.where(
        denominators.gt(0),
        contribution["absolute_effect_sum"] / denominators,
        np.nan,
    )
    contribution["contribution_status"] = np.where(
        denominators.gt(0),
        "DEFINED",
        "ZERO_DENOMINATOR",
    )
    return contribution


def _build_leave_one_game_out(
    observations: pd.DataFrame,
    result_keys: pd.DataFrame,
    *,
    game_ids: Sequence[str] = X13_GAME_IDS,
) -> pd.DataFrame:
    """Apply the registered leave-one-game-out mean to episode-level effects."""

    episode_effects = _episode_level_effects(observations)
    rows: list[dict[str, object]] = []
    for result in result_keys.itertuples(index=False):
        group = episode_effects[
            episode_effects["factor_id"].eq(result.factor_id)
            & episode_effects["market_family"].eq(result.market_family)
            & episode_effects["horizon_seconds"].eq(
                result.horizon_seconds
            )
        ]
        for game_id in game_ids:
            rest = group[~group["game_id"].eq(game_id)]
            rows.append(
                {
                    "factor_id": result.factor_id,
                    "market_family": result.market_family,
                    "horizon_seconds": result.horizon_seconds,
                    "game_id": game_id,
                    "point_estimate": (
                        float(rest["signed_price_change"].mean())
                        if not rest.empty
                        else math.nan
                    ),
                    "episode_count": len(rest),
                }
            )
    return pd.DataFrame(rows)


class ExistingX13StageProviderV2:
    """The one production frame builder for the frozen X-13 discovery set."""

    def __init__(
        self,
        inputs: ExistingX13MaterializationInputsV2,
    ) -> None:
        if not isinstance(inputs, ExistingX13MaterializationInputsV2):
            raise TypeError(
                "inputs must be ExistingX13MaterializationInputsV2"
            )
        self.inputs = inputs
        self._lock = threading.Lock()
        self._feature_frame: pd.DataFrame | None = None
        self._feature_source_hashes: tuple[str, ...] = ()
        self._selected_pbp: pd.DataFrame | None = None
        self._selected_participation: pd.DataFrame | None = None
        self._all_definitions = _definition_records(inputs.verified_inputs)
        (
            self._definitions,
            self._interaction_definitions,
        ) = _partition_registered_factor_definitions(self._all_definitions)
        self._definition_hash_by_id = {
            definition.factor_id: definition.definition_sha256
            for definition in inputs.verified_inputs.factor_definitions
            if ".INTERACTION." not in definition.factor_id
        }
        authority_component_sha256s: dict[str, str] = {}
        for (
            source_name,
            relative,
            _audit_field,
        ) in _AUTHORITY_SOURCE_RELATIVE_PATHS:
            path = inputs.source_artifacts.get(source_name)
            if path is None or not path.is_file():
                raise FactorLabMaterializationError(
                    f"current authority source artifact is missing: "
                    f"{source_name}"
                )
            authority_component_sha256s[relative.as_posix()] = (
                _sha256_path(path)[0]
            )
        self._current_authority_component_sha256s = (
            authority_component_sha256s
        )
        self._current_authority_snapshot_sha256 = canonical_sha256(
            {
                "schema": "x13-current-dataset-authority-v2",
                "components": [
                    {
                        "path": relative.as_posix(),
                        "sha256": authority_component_sha256s[
                            relative.as_posix()
                        ],
                    }
                    for _source_name, relative, _audit_field
                    in _AUTHORITY_SOURCE_RELATIVE_PATHS
                ],
            }
        )

    @property
    def factor_registry(self) -> tuple[dict[str, object], ...]:
        """Return the full review inventory; only base definitions execute."""

        return self._all_definitions

    @property
    def executable_factor_registry(self) -> tuple[dict[str, object], ...]:
        return self._definitions

    @property
    def interaction_factor_inventory(self) -> tuple[dict[str, object], ...]:
        return self._interaction_definitions

    def prepare(self) -> None:
        """Build the corrected PIT view once before any game workers start."""

        with self._lock:
            if self._feature_frame is None:
                published = verify_sports_feature_view(
                    self.inputs.feature_view_manifest_path
                )
                if (
                    published.row_count != _EXPECTED_V2_FEATURE_ROW_COUNT
                    or published.rows_sha256
                    != self.inputs.feature_view_rows_sha256
                    or published.data_path
                    != self.inputs.feature_view_data_path
                    or published.code_lock_sha256
                    != self.inputs.feature_view_code_lock_sha256
                ):
                    raise FactorLabMaterializationError(
                        "published V2 Sports Feature View identity differs"
                    )
                feature_frame = pq.read_table(
                    published.data_path
                ).to_pandas()
                observed_games = tuple(
                    sorted(feature_frame["game_id"].astype(str).unique())
                )
                if (
                    len(feature_frame) != _EXPECTED_V2_FEATURE_ROW_COUNT
                    or observed_games != X13_GAME_IDS
                    or feature_frame.duplicated(["game_id", "play_id"]).any()
                ):
                    raise FactorLabMaterializationError(
                        "corrected Sports Feature View game coverage differs"
                    )
                self._feature_frame = feature_frame
                manifest = _read_canonical_json(
                    published.manifest_path
                )
                manifest_hashes = manifest.get("source_hashes")
                if not isinstance(manifest_hashes, list):
                    raise FactorLabMaterializationError(
                        "published feature source hashes are missing"
                    )
                parsed_hashes = {
                    record.get("sha256")
                    for record in manifest_hashes
                    if isinstance(record, Mapping)
                }
                if (
                    not parsed_hashes
                    or any(
                        not isinstance(value, str)
                        or _SHA256_RE.fullmatch(value) is None
                        for value in parsed_hashes
                    )
                ):
                    raise FactorLabMaterializationError(
                        "published feature source hashes are invalid"
                    )
                self._feature_source_hashes = tuple(
                    sorted(
                        {
                            *parsed_hashes,
                            published.rows_sha256,
                            self.inputs.participation_object_sha256,
                        }
                    )
                )
            if self._selected_pbp is None:
                season_2025 = [
                    path
                    for path in self.inputs.verified_inputs.nflverse_pbp_paths
                    if "partition=season-2025" in str(path)
                ]
                if len(season_2025) != 1:
                    raise FactorLabMaterializationError(
                        "verified 2025 nflverse partition is not unique"
                    )
                schema_names = set(
                    pq.ParquetFile(season_2025[0]).schema_arrow.names
                )
                from prediction_market.sports.nfl_factor_lab_features import (
                    NFL_FACTOR_LAB_PBP_COLUMNS,
                )

                columns = sorted(
                    set(NFL_FACTOR_LAB_PBP_COLUMNS) & schema_names
                )
                selected = pq.read_table(
                    season_2025[0],
                    columns=columns,
                    filters=[
                        (
                            "game_id",
                            "in",
                            list(X13_GAME_IDS),
                        )
                    ],
                ).to_pandas()
                if (
                    len(selected) != 3_699
                    or tuple(
                        sorted(selected["game_id"].astype(str).unique())
                    )
                    != X13_GAME_IDS
                    or selected.duplicated(["game_id", "play_id"]).any()
                ):
                    raise FactorLabMaterializationError(
                        "selected 2025 PBP does not preserve the 3,699-row input"
                    )
                self._selected_pbp = selected
            if self._selected_participation is None:
                participation = pq.read_table(
                    self.inputs.participation_object_path,
                    columns=[
                        "nflverse_game_id",
                        "play_id",
                        "offense_players",
                        "defense_players",
                        "offense_names",
                        "defense_names",
                        "n_offense",
                        "n_defense",
                    ],
                    filters=[
                        (
                            "nflverse_game_id",
                            "in",
                            list(X13_GAME_IDS),
                        )
                    ],
                ).to_pandas()
                participation = participation.rename(
                    columns={"nflverse_game_id": "game_id"}
                )
                participation["play_id"] = participation["play_id"].map(
                    _stable_play_id
                )
                if (
                    len(participation) != 3_425
                    or tuple(
                        sorted(
                            participation["game_id"].astype(str).unique()
                        )
                    )
                    != X13_GAME_IDS
                    or participation.duplicated(["game_id", "play_id"]).any()
                ):
                    raise FactorLabMaterializationError(
                        "verified participation rows do not preserve the "
                        "frozen 20-game coverage"
                    )
                self._selected_participation = participation

    def _game_payload(self, game_id: str) -> dict[str, Any]:
        if game_id not in X13_GAME_IDS:
            raise FactorLabMaterializationError(
                f"game is outside frozen discovery: {game_id}"
            )
        relative = f"games/{game_id}.json"
        path = _verify_file_from_ledger(
            self.inputs.batch_root,
            self.inputs.checksums,
            relative,
        )
        payload = _read_canonical_json(path)
        validated = _validate_game_payload(payload)
        expected = self.inputs.batch_manifest["game_hashes"][game_id]  # type: ignore[index]
        if canonical_payload_hash(validated) != expected:
            raise FactorLabMaterializationError(
                f"game payload changed after input verification: {game_id}"
            )
        return validated

    def _actual_projection(self, game_id: str) -> pd.DataFrame:
        frame = pq.read_table(
            self.inputs.verified_inputs.actual_observations_path,
            filters=[("game_id", "=", game_id)],
        ).to_pandas()
        if frame.empty or set(frame["game_id"]) != {game_id}:
            raise FactorLabMaterializationError(
                f"actual association projection is absent: {game_id}"
            )
        return frame

    def _factor_observation_build(
        self,
        *,
        feature_frame: pd.DataFrame,
        reaction_paths: pd.DataFrame,
        contract_inventory: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        path_contract_columns = {
            "contract_id",
            "family",
            "period",
            "subject",
            "measure",
            "kind",
            "rule_sha256",
        }
        formal_paths = reaction_paths.drop(
            columns=sorted(
                path_contract_columns.intersection(reaction_paths.columns)
            )
        )
        build = build_factor_observations_v2(
            feature_frame,
            formal_paths,
            contract_inventory,
            self._definitions,
        )
        observations = build.factor_observations.copy()
        audit = build.factor_exclusion_audit.copy()
        if audit.empty:
            raise FactorLabMaterializationError(
                "formal V2 factor builder produced no candidate audit"
            )
        if not observations.empty and not observations[
            "factor_definition_sha256"
        ].eq(
            observations["factor_id"].map(self._definition_hash_by_id)
        ).all():
            raise FactorLabMaterializationError(
                "formal V2 factor observation definition identity differs"
            )
        observations["experiment_id"] = "X-13"
        audit["experiment_id"] = "X-13"
        observations["formal_build_sha256"] = (
            build.semantic_rows_sha256
        )
        audit["formal_build_sha256"] = build.semantic_rows_sha256
        return observations, audit

    def _single_game_results(
        self,
        observations: pd.DataFrame,
    ) -> pd.DataFrame:
        keys = [
            "factor_id",
            "factor_version",
            "game_id",
            "market_family",
            "horizon_seconds",
        ]
        if observations.empty:
            return pd.DataFrame(
                columns=[
                    *keys,
                    "observation_count",
                    "episode_count",
                    "point_estimate",
                    "maximum_excursion_mean",
                    "trade_count",
                    "volume",
                    "staleness_seconds_mean",
                    "claim_boundary",
                ]
            )
        results = (
            observations.groupby(keys, sort=True, dropna=False)
            .agg(
                observation_count=("signed_price_change", "size"),
                episode_count=("episode_id", "nunique"),
                point_estimate=("signed_price_change", "mean"),
                maximum_excursion_mean=("maximum_excursion", "mean"),
                trade_count=("trade_count", "sum"),
                volume=("volume", "sum"),
                staleness_seconds_mean=("staleness_seconds", "mean"),
            )
            .reset_index()
        )
        results["claim_boundary"] = (
            "PRELIMINARY_SOURCE_TIME_ONLY; single-game descriptive result"
        )
        return results

    def build_single_game_frames(
        self,
        game_id: str,
        market_game: X13MarketRecleanGameV2,
    ) -> Mapping[str, pd.DataFrame]:
        self.prepare()
        assert self._feature_frame is not None
        assert self._selected_pbp is not None
        assert self._selected_participation is not None
        payload = self._game_payload(game_id)
        game_payload_hash = self.inputs.checksums[f"games/{game_id}.json"]
        selected = self._selected_pbp[
            self._selected_pbp["game_id"].eq(game_id)
        ].copy()
        selected_participation = self._selected_participation[
            self._selected_participation["game_id"].eq(game_id)
        ].copy()
        events = _canonical_events_v2(
            game_id=game_id,
            legacy_events=payload["events"],
            selected_pbp=selected,
            participation=selected_participation,
            participation_source_sha256=(
                self.inputs.participation_object_sha256
            ),
        )
        episodes = pd.DataFrame(payload["episodes"])
        stats = pd.DataFrame(payload["stat_ledger"])
        ledger = build_sports_row_disposition_ledger(
            selected_pbp_rows=selected.to_dict("records"),
            finalized_episode_rows=payload["episodes"],
            canonical_event_rows=payload["events"],
            source_hashes={
                f"x13_source_{index:02d}": digest
                for index, digest in enumerate(self.inputs.source_hashes)
            },
        )
        disposition = pd.DataFrame(ledger.to_records())
        feature = self._feature_frame[
            self._feature_frame["game_id"].eq(game_id)
        ].copy()
        if feature.empty:
            raise FactorLabMaterializationError(
                f"corrected PIT feature rows are absent: {game_id}"
            )
        feature = _pit_features_v2(
            feature_frame=feature,
            selected_pbp=selected,
            participation=selected_participation,
            participation_source_sha256=(
                self.inputs.participation_object_sha256
            ),
        )
        feature["sports_source_hashes"] = [
            self._feature_source_hashes
        ] * len(feature)

        (
            market_frames,
            typed_contracts,
            typed_observations,
            logical_market_by_contract,
        ) = _formal_market_inputs_v2(
            game_id=game_id,
            market_game=market_game,
            current_authority_snapshot_sha256=(
                self._current_authority_snapshot_sha256
            ),
            current_authority_component_sha256s=(
                self._current_authority_component_sha256s
            ),
        )
        contracts = market_frames["contract_inventory"]
        market_observations = market_frames["market_observations"]
        market_audit = market_frames["market_cleaning_audit"]

        typed_ledger = _typed_game_ledger(payload)
        game_intervals = pd.DataFrame(
            build_game_time_intervals_v1(typed_ledger).to_records()
        )
        market_intervals = pd.DataFrame(
            build_market_time_intervals_v1(
                typed_contracts,
                typed_observations,
                game_id=game_id,
            ).to_records()
        )
        timeline, timeline_audit = _build_source_timeline_frames(
            game_id=game_id,
            game_intervals=game_intervals,
            market_intervals=market_intervals,
            canonical_events=events,
            finalized_episodes=episodes,
            market_observations=market_observations,
        )

        actual_projection = self._actual_projection(game_id)
        attrition = _association_attrition(
            actual_projection,
            logical_market_by_contract=logical_market_by_contract,
            market_observations=market_observations,
        )
        attrition["market_capture_sha256"] = market_game.capture_sha256
        current_dataset_registry_sha256 = (
            self._current_authority_component_sha256s[
                "registries/dataset_registry.csv"
            ]
        )
        attrition["current_dataset_registry_sha256"] = (
            current_dataset_registry_sha256
        )
        attrition["market_stage_semantics"] = (
            _MARKET_RECLEAN_STAGE_SEMANTICS
        )
        market_source_hashes = tuple(
            sorted(
                {
                    self.inputs.verified_inputs.signal_bundle_sha256,
                    self.inputs.verified_inputs.cache_sha256,
                    game_payload_hash,
                    market_game.capture_sha256,
                    self._current_authority_snapshot_sha256,
                    *self._current_authority_component_sha256s.values(),
                    *(
                        _sha256_path(
                            self.inputs.source_artifacts[name]
                        )[0]
                        for name in _MARKET_RECLEAN_SOURCE_ARTIFACTS
                    ),
                }
            )
        )
        paths = _build_reaction_paths_v2(
            actual_projection=actual_projection,
            market_observations=market_observations,
            market_audit=market_audit,
            logical_market_by_contract=logical_market_by_contract,
            market_source_hashes=market_source_hashes,
        )
        factor_observations, factor_audit = (
            self._factor_observation_build(
                feature_frame=feature,
                reaction_paths=paths,
                contract_inventory=contracts,
            )
        )
        single_results = self._single_game_results(factor_observations)
        frames = {
            "sports_row_disposition": disposition,
            "canonical_events": events,
            "finalized_episodes": episodes,
            "stat_ledger": stats,
            "pit_features": feature,
            "contract_inventory": contracts,
            "market_observations": market_observations,
            "market_cleaning_audit": market_audit,
            "game_time_intervals": game_intervals,
            "market_time_intervals": market_intervals,
            "source_timeline": timeline,
            "timeline_audit": timeline_audit,
            "association_attrition": attrition,
            "reaction_paths": paths,
            "factor_observations": factor_observations,
            "factor_exclusion_audit": factor_audit,
            "single_game_factor_results": single_results,
        }
        self._validate_single_game_references(
            game_id=game_id,
            frames=frames,
        )
        return {
            name: _arrow_safe_frame(frame)
            for name, frame in frames.items()
        }

    def _validate_single_game_references(
        self,
        *,
        game_id: str,
        frames: Mapping[str, pd.DataFrame],
    ) -> None:
        for name, frame in frames.items():
            if "game_id" not in frame:
                raise FactorLabMaterializationError(
                    f"{name} has no game_id"
                )
            observed = set(frame["game_id"].dropna().astype(str))
            if observed and observed != {game_id}:
                raise FactorLabMaterializationError(
                    f"{name} contains another game"
                )
        events = set(frames["canonical_events"]["play_id"].astype(str))
        dispositions = frames["sports_row_disposition"]
        if not set(
            dispositions["canonical_play_id"].astype(str)
        ).issubset(events):
            raise FactorLabMaterializationError(
                "sports disposition references an unknown canonical play"
            )
        episodes = frames["finalized_episodes"]
        episode_ids = set(episodes["episode_id"].astype(str))
        disposition_episode_ids = set(
            dispositions["episode_id"].dropna().astype(str)
        )
        if not disposition_episode_ids.issubset(episode_ids):
            raise FactorLabMaterializationError(
                "sports disposition references an unknown episode"
            )
        episode_plays = {
            str(play)
            for values in episodes["play_ids"]
            for play in values
        }
        if not episode_plays.issubset(events):
            raise FactorLabMaterializationError(
                "finalized episode references an unknown canonical play"
            )
        if not set(
            frames["stat_ledger"]["through_play_id"].astype(str)
        ).issubset(events):
            raise FactorLabMaterializationError(
                "stat ledger references an unknown canonical play"
            )
        if not set(
            frames["pit_features"]["play_id"].astype(str)
        ).issubset(events):
            raise FactorLabMaterializationError(
                "PIT feature references an unknown canonical play"
            )
        inventory_pairs = set(
            frames["contract_inventory"][
                ["logical_market_id", "outcome"]
            ].itertuples(index=False, name=None)
        )
        market_pairs = set(
            frames["market_observations"][
                ["logical_market_id", "outcome"]
            ].itertuples(index=False, name=None)
        )
        if not market_pairs.issubset(inventory_pairs):
            raise FactorLabMaterializationError(
                "market observation references an unknown contract outcome"
            )
        observation_ids = set(
            frames["market_observations"]["observation_id"].astype(str)
        )
        if not set(
            frames["market_cleaning_audit"]["observation_id"].astype(str)
        ).issubset(observation_ids):
            raise FactorLabMaterializationError(
                "market cleaning audit references an unknown observation"
            )
        if not set(
            frames["game_time_intervals"]["episode_id"].astype(str)
        ).issubset(episode_ids):
            raise FactorLabMaterializationError(
                "Layer G references an unknown episode"
            )
        if not set(
            frames["market_time_intervals"]["observation_id"].astype(str)
        ).issubset(observation_ids):
            raise FactorLabMaterializationError(
                "Layer M references an unknown market observation"
            )
        game_interval_ids = set(
            frames["game_time_intervals"]["interval_id"].astype(str)
        )
        market_interval_ids = set(
            frames["market_time_intervals"]["interval_id"].astype(str)
        )
        timeline = frames["source_timeline"]
        if not set(
            timeline.loc[timeline["layer"].eq("G"), "interval_id"].astype(str)
        ).issubset(game_interval_ids) or not set(
            timeline.loc[timeline["layer"].eq("M"), "interval_id"].astype(str)
        ).issubset(market_interval_ids):
            raise FactorLabMaterializationError(
                "source timeline references an unknown G/M interval"
            )
        timeline_ids = set(timeline["timeline_id"].astype(str))
        if not set(
            frames["timeline_audit"]["timeline_id"].astype(str)
        ).issubset(timeline_ids):
            raise FactorLabMaterializationError(
                "timeline audit references an unknown timeline row"
            )
        contract_ids = set(
            frames["contract_inventory"]["contract_id"].astype(str)
        )
        for stage in (
            "association_attrition",
            "reaction_paths",
            "factor_observations",
        ):
            frame = frames[stage]
            if not set(frame["episode_id"].astype(str)).issubset(episode_ids):
                raise FactorLabMaterializationError(
                    f"{stage} references an unknown episode"
                )
            if "contract_id" in frame and not set(
                frame["contract_id"].astype(str)
            ).issubset(contract_ids):
                raise FactorLabMaterializationError(
                    f"{stage} references an unknown contract"
                )

    def build_cross_game_frames(
        self,
        single_game_bundles: tuple[PublishedSingleGameBundleV2, ...],
    ) -> Mapping[str, pd.DataFrame]:
        if tuple(bundle.game_id for bundle in single_game_bundles) != X13_GAME_IDS:
            raise FactorLabMaterializationError(
                "cross-game builder requires the exact verified single-game set"
            )
        factor_frames: list[pd.DataFrame] = []
        for bundle in single_game_bundles:
            verified = verify_single_game_bundle(bundle.manifest_path)
            factor_frames.append(
                pd.read_parquet(
                    verified.stage_paths["factor_observations"]
                )
            )
        observations = pd.concat(factor_frames, ignore_index=True)
        if set(observations["game_id"].astype(str)) != set(X13_GAME_IDS):
            raise FactorLabMaterializationError(
                "cross-game factor observations do not cover all singles"
            )
        if (
            "market_family" not in observations
            or "family" in observations
        ):
            raise FactorLabMaterializationError(
                "formal factor observations must use market_family only"
            )
        analysis_observations = observations.copy()
        analysis_observations["family"] = analysis_observations[
            "market_family"
        ]
        discovery = run_factor_discovery(
            analysis_observations,
            factor_registry=self._definitions,
            bootstrap_samples=10_000,
            seed=20260725,
        )
        contribution = _build_game_contribution(observations)
        contribution_max = (
            contribution.groupby(
                ["factor_id", "market_family", "horizon_seconds"],
                sort=True,
            )["contribution"]
            .max()
            .rename("materialized_maximum_contribution")
            .reset_index()
        )
        # The registered discovery gate uses 1.0 as a conservative sentinel
        # when every per-game effect sum is zero.  The supporting table keeps
        # the mathematical share null and labels ZERO_DENOMINATOR explicitly.
        contribution_max["materialized_maximum_contribution"] = (
            contribution_max[
                "materialized_maximum_contribution"
            ].fillna(1.0)
        )
        observed_discovery = discovery[
            discovery["observation_count"].gt(0)
        ][
            [
                "factor_id",
                "market_family",
                "horizon_seconds",
                "maximum_single_game_contribution",
            ]
        ].merge(
            contribution_max,
            on=["factor_id", "market_family", "horizon_seconds"],
            how="left",
            validate="one_to_one",
        )
        if (
            observed_discovery[
                "materialized_maximum_contribution"
            ].isna().any()
            or not np.allclose(
                observed_discovery[
                    "maximum_single_game_contribution"
                ].to_numpy(dtype=float),
                observed_discovery[
                    "materialized_maximum_contribution"
                ].to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise FactorLabMaterializationError(
                "game contribution disagrees with discovery maximum"
            )

        result_keys = discovery[
            ["factor_id", "market_family", "horizon_seconds"]
        ].drop_duplicates()
        leave_one_game_out = _build_leave_one_game_out(
            observations,
            result_keys,
        )

        venue = (
            observations.groupby(
                [
                    "factor_id",
                    "market_family",
                    "horizon_seconds",
                    "venue",
                ],
                sort=True,
            )["signed_price_change"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "point_estimate",
                    "count": "observation_count",
                }
            )
        )
        review_rows: list[dict[str, object]] = []
        for definition in self._definitions:
            factor_results = discovery[
                discovery["factor_id"].eq(definition["factor_id"])
            ].sort_values(
                ["market_family", "horizon_seconds"],
                kind="mergesort",
            )
            for result in factor_results.itertuples(index=False):
                review_rows.append(
                    {
                        "factor_id": definition["factor_id"],
                        "market_family": result.market_family,
                        "horizon_seconds": int(result.horizon_seconds),
                        "factor_version": definition["version"],
                        "sports_meaning": definition["sports_meaning"],
                        "predicate": definition["predicate"],
                        "pit_fields": definition["pit_fields"],
                        "exclusions": definition["exclusions"],
                        "target": definition["target"],
                        "review_decision": "PENDING",
                        "discovery_status": result.discovery_status,
                        "support_status": result.support_status,
                        "episode_count": int(result.episode_count),
                        "game_count": int(result.game_count),
                        "observation_count": int(result.observation_count),
                        "point_estimate": result.point_estimate,
                        "ci_low": result.ci_low,
                        "ci_high": result.ci_high,
                        "bh_q_value": result.bh_q_value,
                        "loo_sign_stability_rate": (
                            result.loo_sign_stability_rate
                        ),
                        "maximum_single_game_contribution": (
                            result.maximum_single_game_contribution
                        ),
                        "venue_sign_consistent": (
                            result.venue_sign_consistent
                        ),
                        "holdout_reaction_accessed": False,
                    }
                )
        expert_review = pd.DataFrame(review_rows)
        return {
            "cross_game_factor_results": _arrow_safe_frame(discovery),
            "game_contribution": _arrow_safe_frame(contribution),
            "leave_one_game_out": _arrow_safe_frame(
                leave_one_game_out
            ),
            "venue_consistency": _arrow_safe_frame(venue),
            "expert_review_queue": _arrow_safe_frame(expert_review),
        }


def _require_stage_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    expected: Mapping[str, tuple[str, ...]],
    label: str,
) -> dict[str, pd.DataFrame]:
    if not isinstance(frames, Mapping):
        raise FactorLabMaterializationError(f"{label} must return a mapping")
    missing = set(expected).difference(frames)
    extra = set(frames).difference(expected)
    if missing or extra:
        raise FactorLabMaterializationError(
            f"{label} stage set differs; missing={sorted(missing)!r}; "
            f"extra={sorted(extra)!r}"
        )
    result: dict[str, pd.DataFrame] = {}
    for name in expected:
        frame = frames[name]
        if not isinstance(frame, pd.DataFrame):
            raise FactorLabMaterializationError(
                f"{label}.{name} is not a DataFrame"
            )
        result[name] = frame
    return result


def _validate_cross_game_references(
    frames: Mapping[str, pd.DataFrame],
    *,
    game_ids: tuple[str, ...],
) -> None:
    expected_games = set(game_ids)
    for stage in ("game_contribution", "leave_one_game_out"):
        observed = set(frames[stage]["game_id"].dropna().astype(str))
        if observed != expected_games:
            raise FactorLabMaterializationError(
                f"{stage} does not cover the exact single-game set"
            )
    result_factors = set(
        frames["cross_game_factor_results"]["factor_id"].dropna().astype(str)
    )
    if not result_factors:
        raise FactorLabMaterializationError(
            "cross_game_factor_results contains no factor"
        )
    for stage in (
        "game_contribution",
        "leave_one_game_out",
        "venue_consistency",
        "expert_review_queue",
    ):
        observed = set(frames[stage]["factor_id"].dropna().astype(str))
        if not observed.issubset(result_factors):
            raise FactorLabMaterializationError(
                f"{stage} references a factor absent from cross results"
            )


def materialize_factor_lab_v2(
    plan: FactorLabMaterializationPlanV2,
    *,
    single_game_frame_builder: SingleGameFrameBuilder,
    cross_game_frame_builder: CrossGameFrameBuilder,
) -> MaterializedFactorLabV2:
    """Publish twenty verified singles, then and only then publish cross-game."""

    if not isinstance(plan, FactorLabMaterializationPlanV2):
        raise TypeError("plan must be FactorLabMaterializationPlanV2")
    authoritative_code_lock = factor_lab_materializer_code_lock_sha256(
        plan.project_root
    )
    if plan.code_lock_sha256 != authoritative_code_lock:
        raise FactorLabMaterializationError(
            "plan code lock differs from the authoritative current code lock"
        )
    if plan.game_ids != X13_GAME_IDS:
        raise FactorLabMaterializationError(
            "game_ids must exactly match the frozen X-13 discovery set"
        )
    universe_artifact = plan.source_artifacts[
        "x13_market_universe_v1"
    ]
    capture_pointer = plan.source_artifacts[
        "x13_market_capture_pointer_v2"
    ]
    capture_receipt = plan.source_artifacts[
        "x13_market_capture_receipt_v2"
    ]
    expected_capture_sha256 = (
        "sha256:" + capture_pointer.stem
    )
    expected_receipt_sha256 = (
        "sha256:" + capture_receipt.stem
    )
    if (
        _SHA256_RE.fullmatch(expected_capture_sha256) is None
        or _SHA256_RE.fullmatch(expected_receipt_sha256) is None
        or expected_capture_sha256 not in plan.source_hashes
        or expected_receipt_sha256 not in plan.source_hashes
    ):
        raise FactorLabMaterializationError(
            "market capture pointer/receipt identities are not source-bound"
        )
    raw_store_root = capture_pointer.parent.parent
    market_capture_index = verify_x13_market_capture_v2(
        universe_artifact=universe_artifact,
        capture_pointer=capture_pointer,
        raw_store_root=raw_store_root,
        program_root=plan.project_root,
    )

    def build_and_verify_single(
        game_id: str,
    ) -> PublishedSingleGameBundleV2:
        market_game = reclean_x13_verified_market_game_v2(
            market_capture_index,
            game_id,
        )
        if (
            getattr(market_game, "status", None) != "PASS"
            or getattr(market_game, "game_id", None) != game_id
        ):
            raise FactorLabMaterializationError(
                "indexed reclean game identity or status differs"
            )
        frames = _require_stage_frames(
            single_game_frame_builder(game_id, market_game),
            expected=_SINGLE_STAGE_GRAINS,
            label=f"single_game_frame_builder[{game_id}]",
        )
        published = build_single_game_bundle(
            game_id=game_id,
            stage_frames=frames,
            source_hashes=plan.source_hashes,
            factor_registry_sha256=plan.factor_registry_sha256,
            code_lock_sha256=authoritative_code_lock,
            output_root=plan.single_game_output_root,
        )
        return verify_single_game_bundle(published.manifest_path)

    futures = {}
    with ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix="nfl-x13-single-game",
    ) as executor:
        futures = {
            game_id: executor.submit(build_and_verify_single, game_id)
            for game_id in plan.game_ids
        }
        try:
            singles_by_game = {
                game_id: futures[game_id].result()
                for game_id in plan.game_ids
            }
        except Exception:
            for future in futures.values():
                future.cancel()
            raise

    singles = tuple(singles_by_game[game_id] for game_id in plan.game_ids)
    observed_games = tuple(bundle.game_id for bundle in singles)
    if observed_games != plan.game_ids:
        raise FactorLabMaterializationError(
            "single-game publication order differs from the frozen plan"
        )

    verified_singles = singles
    cross_frames = _require_stage_frames(
        cross_game_frame_builder(verified_singles),
        expected=_CROSS_STAGE_GRAINS,
        label="cross_game_frame_builder",
    )
    _validate_cross_game_references(cross_frames, game_ids=observed_games)
    cross = build_cross_game_bundle(
        single_game_manifests=tuple(
            bundle.manifest_path for bundle in singles
        ),
        cross_game_frames=cross_frames,
        batch_spec_sha256=plan.batch_spec_sha256,
        factor_registry_sha256=plan.factor_registry_sha256,
        code_lock_sha256=authoritative_code_lock,
        output_root=plan.cross_game_output_root,
    )
    verified_cross = verify_cross_game_bundle(cross.manifest_path)
    if verified_cross.game_ids != plan.game_ids:
        raise FactorLabMaterializationError(
            "cross-game bundle does not reference the frozen twenty games"
        )
    run_index = build_factor_lab_run_index(
        single_game_manifests=tuple(
            bundle.manifest_path for bundle in singles
        ),
        cross_game_manifest=verified_cross.manifest_path,
        source_artifacts=plan.source_artifacts,
        project_root=plan.project_root,
        s3_bucket=plan.s3_bucket,
        s3_prefix=plan.s3_prefix,
        output_path=plan.run_index_path,
    )
    return MaterializedFactorLabV2(
        single_game_bundles=singles,
        cross_game_bundle=verified_cross,
        run_index=run_index,
    )


def build_existing_x13_materialization_plan_v2(
    project_root: str | Path,
    *,
    inputs: ExistingX13MaterializationInputsV2 | None = None,
) -> FactorLabMaterializationPlanV2:
    """Create the fixed local/S3 plan for the current frozen X-13 evidence."""

    root = Path(project_root).resolve()
    verified = (
        inputs
        if inputs is not None
        else load_verified_existing_x13_materialization_inputs(root)
    )
    if verified.verified_inputs.project_root.resolve() != root:
        raise FactorLabMaterializationError(
            "verified inputs belong to another project root"
        )
    output_root = (
        root
        / "artifacts/market-observation/nfl/x13/factor-lab/v2"
    )
    return FactorLabMaterializationPlanV2(
        project_root=root,
        single_game_output_root=output_root / "single-game",
        cross_game_output_root=output_root / "cross-game",
        run_index_path=(
            root / "registries/factors/nfl_factor_lab_run_index_v2.json"
        ),
        game_ids=X13_GAME_IDS,
        source_hashes=verified.source_hashes,
        source_artifacts=verified.source_artifacts,
        factor_registry_sha256=(
            verified.verified_inputs.factor_registry_sha256
        ),
        code_lock_sha256=factor_lab_materializer_code_lock_sha256(root),
        batch_spec_sha256=X13_BATCH_SPEC.plan_id,
        s3_bucket="prediction-market-data",
        s3_prefix="prediction-market",
    )


def materialize_existing_x13_factor_lab_v2(
    project_root: str | Path,
) -> MaterializedFactorLabV2:
    """Materialize the verified discovery set; never access holdout reaction."""

    inputs = load_verified_existing_x13_materialization_inputs(project_root)
    provider = ExistingX13StageProviderV2(inputs)
    provider.prepare()
    plan = build_existing_x13_materialization_plan_v2(
        project_root,
        inputs=inputs,
    )
    return materialize_factor_lab_v2(
        plan,
        single_game_frame_builder=provider.build_single_game_frames,
        cross_game_frame_builder=provider.build_cross_game_frames,
    )


def mirror_materialized_factor_lab_v2(
    materialized: MaterializedFactorLabV2,
    *,
    project_root: str | Path,
    s3_bucket: str,
    s3_prefix: str,
    object_store: ImmutableObjectStore,
) -> int:
    """Mirror a completed local materialization; no upstream fallback exists."""

    if not isinstance(materialized, MaterializedFactorLabV2):
        raise TypeError("materialized must be MaterializedFactorLabV2")
    count = 0
    for bundle in (
        *materialized.single_game_bundles,
        materialized.cross_game_bundle,
    ):
        count += len(
            mirror_bundle(
                bundle,
                project_root=project_root,
                s3_bucket=s3_bucket,
                s3_prefix=s3_prefix,
                object_store=object_store,
            )
        )
    return count
