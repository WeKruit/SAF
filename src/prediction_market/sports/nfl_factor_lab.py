"""Public, governed interfaces for the NFL Sports Factor Lab.

The workbench and Quarto reports import this module.  It binds them to the
same immutable X-13 inputs and keeps notebooks from becoming a second
cleaning or statistics implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import pandas as pd
import pyarrow.parquet as pq

from prediction_market.sports.nfl_factor_lab_features import (
    NFL_FACTOR_LAB_PBP_COLUMNS,
)
from prediction_market.sports.nfl_factor_lab_analysis import (
    HumanFactorReviewV1,
    ShortlistLockV1,
    build_factor_observations,
    freeze_factor_shortlist as _freeze_factor_shortlist,
    run_factor_discovery as _run_factor_discovery,
    run_locked_historical_holdout as _run_locked_historical_holdout,
)
from prediction_market.sports.nfl_factor_lab_types import (
    FactorDefinitionV1,
    FactorObservationV1,
    FactorResultV1,
    FrontendReferenceV1,
    PredicateClauseV1,
    SportsFeatureViewV1,
)
from prediction_market.sports.nfl_factor_lab_bundles import (
    build_cross_game_bundle,
    build_single_game_bundle,
)
from prediction_market.sports.nfl_x13_game_build import (
    verify_x13_game_state_publication,
)
from prediction_market.sports.nfl_x13_signal_exploration import (
    verify_signal_exploration_bundle,
)


_SIGNAL_BUNDLE_SHA256 = (
    "sha256:9e4afe95b6c9dd500cfd10221575e8a546592485eb5829d6027b61ad38b397be"
)
_CACHE_SHA256 = (
    "sha256:48d1f35e0e7fc63dc1a75611da6e81b2834c69d2d7fe6be4af4d9e3074e6d62e"
)
_HOLDOUT_SELECTION_SHA256 = (
    "sha256:6ce0c257106fa8badd9b43d7a5615fa582ce8d0eae714bdf1453334116716294"
)
_GAME_STATE_BUILD_SHA256 = (
    "sha256:74feebbc358d661118812e34e30f1b4f5a38f3eca377a779392fcab7dc970ef8"
)
_FACTOR_REGISTRY_FILE_SHA256 = (
    "sha256:112fb9cc49d973c160579befdcd1835603b75090e1f6cb1cd39df9b89560567d"
)
_DASHBOARD_BASE_URL = (
    "https://saf-x13-research.windy-crab-9030.chatgpt.site/"
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/ -]{0,511}$")
_DELAYS = frozenset({0, 1, 2, 3, 5, 10})
_HORIZONS = frozenset({1, 2, 5, 10, 30, 60})
class FactorLabInterfaceError(ValueError):
    """A public Factor Lab interface cannot satisfy its frozen contract."""


@dataclass(frozen=True, slots=True)
class VerifiedX13InputsV1:
    experiment_id: str
    claim_boundary: str
    project_root: Path
    data_checkout_root: Path
    signal_manifest_path: Path
    signal_bundle_sha256: str
    cache_manifest_path: Path
    cache_sha256: str
    actual_observations_path: Path
    actual_observation_paths_path: Path
    observed_cache_path: Path
    factor_registry_path: Path
    factor_registry_sha256: str
    factor_definitions: tuple[FactorDefinitionV1, ...]
    method_catalog_path: Path
    method_catalog_sha256: str
    game_state_root: Path
    game_state_build_sha256: str
    holdout_selection_path: Path
    holdout_selection_id: str
    discovery_game_ids: tuple[str, ...]
    holdout_game_ids: tuple[str, ...]
    nflverse_pbp_paths: tuple[Path, ...]
    source_hashes: tuple[str, ...]
    holdout_reaction_accessed: bool = False

    @property
    def discovery_game_count(self) -> int:
        return len(self.discovery_game_ids)

    @property
    def holdout_game_count(self) -> int:
        return len(self.holdout_game_ids)

    def __repr__(self) -> str:
        return (
            "VerifiedX13InputsV1("
            f"discovery_games={self.discovery_game_count}, "
            f"holdout_games={self.holdout_game_count}, "
            f"factors={len(self.factor_definitions)}, "
            f"claim_boundary={self.claim_boundary!r})"
        )


@dataclass(frozen=True, slots=True)
class QuartoRenderV1:
    output_path: Path
    quarto_version: str
    parameters: dict[str, object]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _strict_json(path: Path, *, newline: bool | None = None) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FactorLabInterfaceError(f"cannot read strict JSON: {path}") from error
    if type(value) is not dict:
        raise FactorLabInterfaceError(f"JSON root must be an object: {path}")
    canonical = _canonical_bytes(value)
    if newline is True:
        canonical += b"\n"
    elif newline is None and raw.endswith(b"\n"):
        canonical += b"\n"
    if raw != canonical:
        raise FactorLabInterfaceError(f"JSON is not canonical: {path}")
    return value


def _data_checkout_root(project_root: Path) -> Path:
    if project_root.parent.name == ".worktrees":
        candidate = project_root.parent.parent
        if (candidate / ".git").is_dir():
            return candidate
    return project_root


def _factor_definitions(
    path: Path,
) -> tuple[dict[str, Any], tuple[FactorDefinitionV1, ...]]:
    raw = path.read_bytes()
    observed_sha = _sha256_bytes(raw)
    if observed_sha != _FACTOR_REGISTRY_FILE_SHA256:
        raise FactorLabInterfaceError(
            "factor registry hash differs from the X-13 factor-lab lock"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FactorLabInterfaceError("factor registry is invalid JSON") from error
    if (
        type(payload) is not dict
        or payload.get("schema") != "nfl_factor_registry_v1"
        or payload.get("experiment_id") != "X-13"
        or payload.get("status") != "PRELIMINARY_SOURCE_TIME_ONLY"
        or payload.get("bootstrap_samples") != 10_000
    ):
        raise FactorLabInterfaceError("factor registry header is invalid")
    records = payload.get("factor_definitions")
    if type(records) is not list or not records:
        raise FactorLabInterfaceError("factor registry has no definitions")
    definitions: list[FactorDefinitionV1] = []
    observed_ids: list[str] = []
    for record in records:
        if type(record) is not dict:
            raise FactorLabInterfaceError("factor definition is not an object")
        try:
            predicates = tuple(
                sorted(
                    (
                        PredicateClauseV1(
                            field=clause["field"],
                            operator=clause["operator"],
                            value=(
                                tuple(clause["value"])
                                if clause["operator"] in {"in", "not_in"}
                                else clause["value"]
                            ),
                        )
                        for clause in record["predicate"]
                    ),
                    key=lambda item: (item.field, item.operator, repr(item.value)),
                )
            )
            exclusions = tuple(
                sorted(
                    (
                        PredicateClauseV1(
                            field=clause["field"],
                            operator=clause["operator"],
                            value=(
                                tuple(clause["value"])
                                if clause["operator"] in {"in", "not_in"}
                                else clause["value"]
                            ),
                        )
                        for clause in record["exclusions"]
                    ),
                    key=lambda item: (item.field, item.operator, repr(item.value)),
                )
            )
            definition = FactorDefinitionV1(
                factor_id=record["factor_id"],
                version=record["version"],
                sports_meaning=record["sports_meaning"],
                predicate=predicates,
                pit_fields=tuple(record["pit_fields"]),
                focal_entity=record["focal_entity"],
                exclusions=exclusions,
                target=record["target"],
                market_families=tuple(record["market_families"]),
                horizons_seconds=tuple(record["horizons_seconds"]),
                support_kind=record["support_kind"],
                minimum_episodes=record["minimum_episodes"],
                minimum_games=record["minimum_games"],
                minimum_episodes_per_venue=record[
                    "minimum_episodes_per_venue"
                ],
                claim_boundary=record["claim_boundary"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FactorLabInterfaceError(
                f"factor definition failed closed: {record.get('factor_id')}"
            ) from error
        definitions.append(definition)
        observed_ids.append(definition.factor_id)
    if observed_ids != sorted(set(observed_ids)):
        raise FactorLabInterfaceError(
            "factor definitions must be sorted and unique"
        )
    by_id = {item.factor_id: item for item in definitions}
    templates = payload.get("interaction_templates")
    if type(templates) is not list or not templates:
        raise FactorLabInterfaceError("factor interaction templates are missing")
    expanded: list[FactorDefinitionV1] = list(definitions)
    for template in templates:
        if (
            type(template) is not dict
            or template.get("maximum_order") != 2
            or template.get("claim_boundary")
            != "PRELIMINARY_SOURCE_TIME_ONLY"
        ):
            raise FactorLabInterfaceError("interaction template is invalid")
        event_ids = template.get("event_factor_ids")
        prefixes = template.get("moderator_prefixes")
        if (
            type(event_ids) is not list
            or type(prefixes) is not list
            or not event_ids
            or not prefixes
        ):
            raise FactorLabInterfaceError(
                "interaction template universe is empty"
            )
        moderators = [
            item
            for item in definitions
            if any(item.factor_id.startswith(prefix) for prefix in prefixes)
        ]
        if not moderators:
            raise FactorLabInterfaceError(
                "interaction template has no registered moderators"
            )
        for event_id in event_ids:
            event = by_id.get(event_id)
            if event is None:
                raise FactorLabInterfaceError(
                    f"interaction event is unregistered: {event_id}"
                )
            for moderator in moderators:
                clauses = tuple(
                    sorted(
                        set((*event.predicate, *moderator.predicate)),
                        key=lambda item: (
                            item.field,
                            item.operator,
                            repr(item.value),
                        ),
                    )
                )
                exclusions = tuple(
                    sorted(
                        set((*event.exclusions, *moderator.exclusions)),
                        key=lambda item: (
                            item.field,
                            item.operator,
                            repr(item.value),
                        ),
                    )
                )
                suffix = (
                    event.factor_id.removeprefix("NFL.").replace(".", "_")
                    + "__"
                    + moderator.factor_id.removeprefix("NFL.").replace(
                        ".", "_"
                    )
                )
                families = tuple(
                    sorted(
                        set(event.market_families)
                        & set(moderator.market_families)
                    )
                )
                if not families:
                    raise FactorLabInterfaceError(
                        "interaction has no common market family"
                    )
                expanded.append(
                    FactorDefinitionV1(
                        factor_id=f"NFL.INTERACTION.{suffix}",
                        version="v1",
                        sports_meaning=(
                            f"{event.sports_meaning} Moderator: "
                            f"{moderator.sports_meaning}"
                        ),
                        predicate=clauses,
                        pit_fields=tuple(
                            sorted(
                                set(event.pit_fields)
                                | set(moderator.pit_fields)
                            )
                        ),
                        focal_entity=event.focal_entity,
                        exclusions=exclusions,
                        target=event.target,
                        market_families=families,
                        horizons_seconds=tuple(
                            template["horizons_seconds"]
                        ),
                        support_kind=template["support_kind"],
                        minimum_episodes=template["minimum_episodes"],
                        minimum_games=template["minimum_games"],
                        minimum_episodes_per_venue=template[
                            "minimum_episodes_per_venue"
                        ],
                        claim_boundary=template["claim_boundary"],
                    )
                )
    expanded.sort(key=lambda item: item.factor_id)
    if len({item.factor_id for item in expanded}) != len(expanded):
        raise FactorLabInterfaceError("expanded factor IDs are not unique")
    return payload, tuple(expanded)


def _signal_file_map(
    manifest_path: Path, manifest: Mapping[str, object]
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    files = manifest.get("files")
    if type(files) is not list:
        raise FactorLabInterfaceError("signal manifest has no file inventory")
    for record in files:
        if type(record) is not dict:
            raise FactorLabInterfaceError("signal file descriptor is invalid")
        name = record.get("logical_name")
        relative = record.get("path")
        if type(name) is not str or type(relative) is not str:
            raise FactorLabInterfaceError("signal file descriptor is incomplete")
        result[name] = manifest_path.parent / relative
    return result


def _verify_cache(cache_root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = cache_root / "manifest.json"
    # This immutable v1 writer used canonical compact JSON plus a trailing
    # newline.  Accept that frozen byte convention; do not rewrite the object.
    manifest = _strict_json(manifest_path, newline=None)
    if (
        manifest.get("schema") != "nfl_x13_correlation_cache_manifest_v1"
        or manifest.get("experiment_id") != "X-13"
        or manifest.get("cache_id") != _CACHE_SHA256
        or cache_root.name
        != f"sha256-{_CACHE_SHA256.removeprefix('sha256:')}"
    ):
        raise FactorLabInterfaceError("correlation cache identity is invalid")
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict:
        raise FactorLabInterfaceError("cache manifest has no artifacts")
    paths: dict[str, Path] = {}
    for role, descriptor in artifacts.items():
        if type(role) is not str or type(descriptor) is not dict:
            raise FactorLabInterfaceError("cache artifact descriptor is invalid")
        relative = descriptor.get("relative_path")
        expected_sha = descriptor.get("sha256")
        byte_length = descriptor.get("byte_length")
        row_count = descriptor.get("row_count")
        if (
            type(relative) is not str
            or type(expected_sha) is not str
            or _SHA256_RE.fullmatch(expected_sha) is None
            or type(byte_length) is not int
            or type(row_count) is not int
        ):
            raise FactorLabInterfaceError("cache artifact descriptor is incomplete")
        path = cache_root / relative
        if not path.is_file() or path.stat().st_size != byte_length:
            raise FactorLabInterfaceError("cache artifact length differs")
        if _sha256_file(path) != expected_sha:
            raise FactorLabInterfaceError("cache artifact hash differs")
        if path.suffix == ".parquet" and pq.ParquetFile(path).metadata.num_rows != row_count:
            raise FactorLabInterfaceError("cache Parquet row count differs")
        paths[role] = path
    return manifest, paths


def _verify_holdout(path: Path) -> dict[str, Any]:
    value = _strict_json(path, newline=None)
    if (
        value.get("artifact_type")
        != "nfl_x13_historical_holdout_selection"
        or value.get("status") != "FROZEN"
        or value.get("seed") != 20260725
        or value.get("selection_id") != _HOLDOUT_SELECTION_SHA256
        or path.parent.name
        != f"sha256-{_HOLDOUT_SELECTION_SHA256.removeprefix('sha256:')}"
    ):
        raise FactorLabInterfaceError("historical holdout lock is invalid")
    material = {
        "status": value.get("status"),
        "seed": value.get("seed"),
        "selected_game_ids": value.get("selected_game_ids"),
        "evidence_index_sha256": value.get("evidence_index_sha256"),
        "evidence_manifest_sha256s": value.get(
            "evidence_manifest_sha256s"
        ),
    }
    if _sha256_bytes(_canonical_bytes(material)) != _HOLDOUT_SELECTION_SHA256:
        raise FactorLabInterfaceError("historical holdout semantic hash differs")
    payload = path.read_bytes()
    checksum = path.parent / "checksums.sha256"
    expected = (
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n"
    ).encode("ascii")
    if not checksum.is_file() or checksum.read_bytes() != expected:
        raise FactorLabInterfaceError("historical holdout checksum differs")
    return value


def _method_catalog(path: Path) -> str:
    value = json.loads(path.read_bytes())
    if (
        type(value) is not dict
        or value.get("schema_version") != "MethodCatalogV1"
        or type(value.get("catalog_sha256")) is not str
    ):
        raise FactorLabInterfaceError("method catalog is invalid")
    # The catalog's dedicated loader and tests validate per-card semantic hashes;
    # this interface also binds the exact byte object used by the workbench.
    return _sha256_file(path)


def verify_and_load_x13_inputs(
    project_root: str | Path | None = None,
) -> VerifiedX13InputsV1:
    """Verify and bind the small X-13 cache/bundle inputs without replaying."""

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    signal_root = (
        root
        / "artifacts/market-observation/nfl/x13/signal-exploration"
        / f"sha256-{_SIGNAL_BUNDLE_SHA256.removeprefix('sha256:')}"
    )
    signal_manifest_path = signal_root / (
        f"{_SIGNAL_BUNDLE_SHA256.removeprefix('sha256:')}.manifest.json"
    )
    try:
        signal_manifest = verify_signal_exploration_bundle(signal_manifest_path)
    except Exception as error:
        raise FactorLabInterfaceError(
            "verified X-13 signal bundle is unavailable"
        ) from error
    if signal_manifest.get("bundle_sha256") != _SIGNAL_BUNDLE_SHA256:
        raise FactorLabInterfaceError("X-13 signal bundle is not the frozen input")
    signal_files = _signal_file_map(signal_manifest_path, signal_manifest)

    cache_root = (
        root
        / "artifacts/market-observation/nfl/x13/correlation-cache"
        / f"sha256-{_CACHE_SHA256.removeprefix('sha256:')}"
    )
    cache_manifest, cache_files = _verify_cache(cache_root)

    factor_registry_path = root / "registries/factors/nfl_factor_registry_v1.json"
    _, factor_definitions = _factor_definitions(factor_registry_path)
    method_catalog_path = root / "registries/methods/method_catalog_v1.json"
    method_catalog_sha256 = _method_catalog(method_catalog_path)

    game_state_root = (
        root
        / "artifacts/market-observation/nfl/x13/game-state"
        / f"sha256-{_GAME_STATE_BUILD_SHA256.removeprefix('sha256:')}"
    )
    game_state_manifest = _strict_json(
        game_state_root / "build_manifest.json", newline=None
    )
    if (
        game_state_manifest.get("build_id") != _GAME_STATE_BUILD_SHA256
        or game_state_manifest.get("game_count") != 20
        or game_state_manifest.get("deterministic_two_run_hashes_equal")
        is not True
    ):
        raise FactorLabInterfaceError("game-state build is not frozen and deterministic")
    try:
        verify_x13_game_state_publication(game_state_root)
    except Exception as error:
        raise FactorLabInterfaceError(
            "game-state publication checksum verification failed"
        ) from error

    holdout_selection_path = (
        root
        / "artifacts/market-observation/nfl/x13/historical-holdout"
        / f"sha256-{_HOLDOUT_SELECTION_SHA256.removeprefix('sha256:')}"
        / (
            f"{_HOLDOUT_SELECTION_SHA256.removeprefix('sha256:')}"
            ".holdout-selection.json"
        )
    )
    holdout = _verify_holdout(holdout_selection_path)
    holdout_games = tuple(holdout["selected_game_ids"])
    signal_holdout = signal_manifest.get("historical_holdout_selection")
    if (
        type(signal_holdout) is not dict
        or tuple(signal_holdout.get("selected_game_ids", ())) != holdout_games
    ):
        raise FactorLabInterfaceError("signal bundle and holdout lock differ")

    report = json.loads(signal_files["report"].read_bytes())
    discovery_games = tuple(report["selected_game_ids"])
    if (
        len(discovery_games) != 20
        or len(set(discovery_games)) != 20
        or set(discovery_games) & set(holdout_games)
    ):
        raise FactorLabInterfaceError("discovery/holdout game split is invalid")

    data_root = _data_checkout_root(root)
    pbp_paths = tuple(
        sorted(
            (
                data_root
                / "var/raw/raw/source=nflverse/dataset=DS-NFLVERSE"
            ).glob("version=*/partition=season-202[2-5]/*.parquet")
        )
    )
    if len(pbp_paths) != 4:
        raise FactorLabInterfaceError(
            "frozen nflverse 2022-2025 Parquet set is incomplete"
        )
    for path in pbp_paths:
        expected = "sha256:" + path.stem
        if _sha256_file(path) != expected:
            raise FactorLabInterfaceError(
                f"nflverse content hash differs from object name: {path}"
            )
    source_hashes = tuple(
        sorted(
            {
                _SIGNAL_BUNDLE_SHA256,
                _CACHE_SHA256,
                _HOLDOUT_SELECTION_SHA256,
                _GAME_STATE_BUILD_SHA256,
                _FACTOR_REGISTRY_FILE_SHA256,
                method_catalog_sha256,
                *(_sha256_file(path) for path in pbp_paths),
            }
        )
    )
    return VerifiedX13InputsV1(
        experiment_id="X-13",
        claim_boundary="PRELIMINARY_SOURCE_TIME_ONLY",
        project_root=root,
        data_checkout_root=data_root,
        signal_manifest_path=signal_manifest_path,
        signal_bundle_sha256=_SIGNAL_BUNDLE_SHA256,
        cache_manifest_path=cache_root / "manifest.json",
        cache_sha256=cache_manifest["cache_id"],
        actual_observations_path=signal_files["actual_observations"],
        actual_observation_paths_path=signal_files[
            "actual_observation_paths"
        ],
        observed_cache_path=cache_files["observed_cache"],
        factor_registry_path=factor_registry_path,
        factor_registry_sha256=_FACTOR_REGISTRY_FILE_SHA256,
        factor_definitions=factor_definitions,
        method_catalog_path=method_catalog_path,
        method_catalog_sha256=method_catalog_sha256,
        game_state_root=game_state_root,
        game_state_build_sha256=_GAME_STATE_BUILD_SHA256,
        holdout_selection_path=holdout_selection_path,
        holdout_selection_id=_HOLDOUT_SELECTION_SHA256,
        discovery_game_ids=discovery_games,
        holdout_game_ids=holdout_games,
        nflverse_pbp_paths=pbp_paths,
        source_hashes=source_hashes,
        holdout_reaction_accessed=False,
    )


def _stable_reference(value: str, field: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise FactorLabInterfaceError(f"{field} is not a stable reference")
    return value


def build_dashboard_url(
    *,
    game_id: str,
    episode_id: str,
    market: str,
    delay_s: int,
    horizon_s: int,
    base_url: str = _DASHBOARD_BASE_URL,
) -> str:
    if type(delay_s) is not int or delay_s not in _DELAYS:
        raise FactorLabInterfaceError("delay is not registered")
    if type(horizon_s) is not int or horizon_s not in _HORIZONS:
        raise FactorLabInterfaceError("horizon is not registered")
    if not base_url.startswith("https://"):
        raise FactorLabInterfaceError("dashboard base URL must be HTTPS")
    query = urlencode(
        {
            "game": _stable_reference(game_id, "game_id"),
            "episode": _stable_reference(episode_id, "episode_id"),
            "market": _stable_reference(market, "market"),
            "delay_s": delay_s,
            "horizon_s": horizon_s,
        }
    )
    return f"{base_url.rstrip('/')}/?{query}"


def build_sports_feature_view(
    inputs: VerifiedX13InputsV1,
    **kwargs: object,
) -> SportsFeatureViewV1:
    """Build the formal view from verified local immutable inputs.

    Tests and specialized runners may provide explicit verified row
    collections.  The public no-argument path reads only the bounded PBP
    columns needed by the formal builder and never scans market associations.
    """

    if not isinstance(inputs, VerifiedX13InputsV1):
        raise FactorLabInterfaceError("inputs must be VerifiedX13InputsV1")
    from prediction_market.sports.nfl_factor_lab_features import (
        build_sports_feature_view as pure_builder,
    )
    from prediction_market.sports.nfl_x13_spec import X13_GAME_BINDINGS

    if not kwargs:
        selected_rows: list[dict[str, object]] = []
        historical_rows: list[dict[str, object]] = []
        for path in inputs.nflverse_pbp_paths:
            schema_names = set(pq.ParquetFile(path).schema_arrow.names)
            columns = sorted(
                set(NFL_FACTOR_LAB_PBP_COLUMNS) & schema_names
            )
            historical_rows.extend(
                pq.read_table(path, columns=columns).to_pylist()
            )
            if "partition=season-2025" in str(path):
                selected_rows.extend(
                    pq.read_table(
                        path,
                        columns=columns,
                        filters=[
                            (
                                "game_id",
                                "in",
                                list(inputs.discovery_game_ids),
                            )
                        ],
                    ).to_pylist()
                )
        observed_games = {
            str(row["game_id"]) for row in selected_rows
        }
        if observed_games != set(inputs.discovery_game_ids):
            raise FactorLabInterfaceError(
                "selected nflverse rows do not exactly cover discovery games"
            )
        episode_rows: list[dict[str, object]] = []
        canonical_rows: list[dict[str, object]] = []
        for game_id in inputs.discovery_game_ids:
            game_root = inputs.game_state_root / "games" / game_id
            episodes = json.loads(
                (game_root / "finalized_episodes.json").read_bytes()
            )
            events = json.loads(
                (game_root / "canonical_events.json").read_bytes()
            )
            if type(episodes) is not list or type(events) is not list:
                raise FactorLabInterfaceError(
                    "game-state rows are not canonical arrays"
                )
            episode_rows.extend(episodes)
            canonical_rows.extend(events)
        kickoff_times: dict[str, str] = {}
        for game_id in inputs.discovery_game_ids:
            times = sorted(
                str(row["time_of_day"])
                for row in selected_rows
                if row.get("game_id") == game_id
                and row.get("time_of_day") is not None
            )
            if not times:
                raise FactorLabInterfaceError(
                    f"discovery game has no source timestamp: {game_id}"
                )
            kickoff_times[game_id] = times[0]
        kwargs = {
            "selected_pbp_rows": selected_rows,
            "finalized_episode_rows": episode_rows,
            "historical_pbp_rows": historical_rows,
            "kickoff_times_utc": kickoff_times,
            "canonical_event_rows": canonical_rows,
            "frozen_game_bindings": X13_GAME_BINDINGS,
        }
    kwargs.setdefault(
        "source_hashes",
        {
            f"x13_source_{index:02d}": digest
            for index, digest in enumerate(inputs.source_hashes)
        },
    )
    return pure_builder(**kwargs)  # type: ignore[arg-type]


def _definition_records(
    definitions: Sequence[FactorDefinitionV1],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for definition in definitions:
        value = definition.to_dict()
        value.pop("schema_version")
        records.append(value)
    return records


def run_factor_discovery(
    inputs: VerifiedX13InputsV1,
    feature_view: SportsFeatureViewV1 | None = None,
    *,
    observations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if feature_view is None:
        raise FactorLabInterfaceError("formal discovery requires SportsFeatureViewV1")
    market = (
        observations
        if observations is not None
        else pd.read_parquet(
            inputs.actual_observations_path,
            filters=[("actual_observation", "=", True)],
        )
    )
    registry = _definition_records(inputs.factor_definitions)
    factor_rows = build_factor_observations(
        pd.DataFrame(feature_view.to_records()),
        market,
        registry,
    )
    return _run_factor_discovery(
        factor_rows,
        factor_registry=registry,
        bootstrap_samples=10_000,
        seed=20260725,
    )


def freeze_factor_shortlist(
    inputs: VerifiedX13InputsV1,
    discovery: pd.DataFrame,
    reviews: Sequence[HumanFactorReviewV1],
    *,
    output_root: str | Path | None = None,
) -> ShortlistLockV1:
    return _freeze_factor_shortlist(
        discovery_results=discovery,
        factor_registry=_definition_records(inputs.factor_definitions),
        reviews=reviews,
        source_hashes=inputs.source_hashes,
        output_root=(
            Path(output_root)
            if output_root is not None
            else inputs.project_root
            / "artifacts/market-observation/nfl/x13/factor-shortlists"
        ),
    )


def run_locked_historical_holdout(
    inputs: VerifiedX13InputsV1,
    shortlist: ShortlistLockV1,
    *,
    holdout_observations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if holdout_observations is None:
        raise FactorLabInterfaceError(
            "holdout reaction bundle is not captured; shortlist lock alone "
            "cannot authorize reaction access"
        )
    return _run_locked_historical_holdout(
        shortlist_manifest=shortlist.manifest_path,
        holdout_observations=holdout_observations,
        factor_registry=_definition_records(inputs.factor_definitions),
        holdout_selection_id=inputs.holdout_selection_id,
        holdout_game_ids=inputs.holdout_game_ids,
        discovery_game_ids=inputs.discovery_game_ids,
    )


def render_quarto_reports(
    inputs: VerifiedX13InputsV1,
    *,
    output_root: str | Path,
    game_id: str | None = None,
    factor_id: str | None = None,
    market_family: str = "moneyline",
    venue: str = "all",
    horizon: int = 10,
    review_status: str = "all",
) -> QuartoRenderV1:
    executable = os.environ.get("SAF_QUARTO") or shutil.which("quarto")
    if executable is None:
        raise FactorLabInterfaceError(
            "Quarto 1.9.38 is required; no report was fabricated"
        )
    version = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != "1.9.38":
        raise FactorLabInterfaceError(
            f"Quarto 1.9.38 is required; observed {version!r}"
        )
    if horizon not in _HORIZONS:
        raise FactorLabInterfaceError("horizon is not registered")
    parameters: dict[str, object] = {
        "game_id": game_id,
        "factor_id": factor_id,
        "market_family": market_family,
        "venue": venue,
        "horizon": horizon,
        "review_status": review_status,
    }
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "render",
        str(
            inputs.project_root
            / "reports/nfl-factor-lab"
        ),
        "--output-dir",
        str(output),
    ]
    for name, value in parameters.items():
        encoded = "null" if value is None else str(value)
        command.extend(["-P", f"{name}:{encoded}"])
    subprocess.run(
        command,
        check=True,
        cwd=inputs.project_root,
        env={**os.environ, "PYTHONPATH": str(inputs.project_root / "src")},
    )
    generated = output / "factor-lab.html"
    if not generated.is_file():
        raise FactorLabInterfaceError("Quarto completed without the expected HTML")
    return QuartoRenderV1(
        output_path=generated,
        quarto_version=version,
        parameters=parameters,
    )


__all__ = [
    "FactorObservationV1",
    "FactorResultV1",
    "FactorLabInterfaceError",
    "FrontendReferenceV1",
    "NFL_FACTOR_LAB_PBP_COLUMNS",
    "QuartoRenderV1",
    "VerifiedX13InputsV1",
    "build_cross_game_bundle",
    "build_dashboard_url",
    "build_single_game_bundle",
    "build_sports_feature_view",
    "freeze_factor_shortlist",
    "render_quarto_reports",
    "run_factor_discovery",
    "run_locked_historical_holdout",
    "verify_and_load_x13_inputs",
]
