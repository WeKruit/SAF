"""Build the corrected regulation-only NFL X-15 Stage-B decision panel.

The builder consumes four already-published sources and nothing else:

* Exact-153 Facts V4;
* FinalizedEpisodeV1;
* EventPrestateContextV1; and
* the exact-153 normalized historical market publication.

It never fetches upstream data and never reads the final holdout.  Each
development game is built and published independently before a small batch
index is published.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
from typing import Any, Final, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.research.nfl_x15_development_panel import (
    DevelopmentPanelError,
    DevelopmentSourceSpec,
    VerifiedDevelopmentSources,
    _adapt_market,
    _read_verified_table,
    _table_descriptor,
    _verify_game_manifest,
    default_development_source_spec,
    verify_development_sources,
)


SCHEMA_VERSION: Final[str] = "RegulationDecisionPanelV3"
ATTRITION_SCHEMA: Final[str] = "RegulationDecisionPanelAttritionV1"
GAME_MANIFEST_SCHEMA: Final[str] = "RegulationDecisionPanelGameManifestV1"
BATCH_MANIFEST_SCHEMA: Final[str] = "RegulationDecisionPanelBatchManifestV1"
BUILDER_VERSION: Final[str] = "nfl-x15-regulation-decision-panel-v3"
EXPERIMENT_ID: Final[str] = "X-15"
DATASET_SPLIT: Final[str] = "DEVELOPMENT"
CLAIM_BOUNDARY: Final[str] = (
    "HISTORICAL_TRADES_ONLY_REGULATION_SOURCE_TIME_PROBABILITY;"
    "NO_LIVE_LATENCY;NO_CAUSALITY;NO_EXECUTION;NO_ALPHA_CLAIM"
)
DIRECTION_THRESHOLD: Final[float] = 0.01
LANDMARK_SECONDS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10)
ENDPOINT_SECONDS: Final[tuple[int, ...]] = tuple(range(5, 61, 5))
EXPECTED_GAME_COUNT: Final[int] = 153

DEFAULT_FINALIZED_EPISODE_MANIFEST: Final[Path] = Path(
    "artifacts/market-observation/nfl/x15/"
    "finalized-episodes-regulation-v1/manifests/sha256/02/"
    "02171082e07cf5262ccb9b0033738977070008d5bdc25d064e4944374f04c95f"
    ".manifest.json"
)
DEFAULT_FINALIZED_EPISODE_MANIFEST_SHA256: Final[str] = (
    "sha256:02171082e07cf5262ccb9b0033738977070008d5bdc25d064e4944374f04c95f"
)
DEFAULT_CONTEXT_MANIFEST: Final[Path] = Path(
    "artifacts/market-observation/nfl/x15/"
    "event-prestate-context-v1/manifests/sha256/24/"
    "24a1427e5e3b032efe2d29da4f4b3a2d975a44fab2953572ba071ffaabb2b51b"
    ".manifest.json"
)
DEFAULT_CONTEXT_MANIFEST_SHA256: Final[str] = (
    "sha256:24a1427e5e3b032efe2d29da4f4b3a2d975a44fab2953572ba071ffaabb2b51b"
)
DEFAULT_OUTPUT_ROOT: Final[Path] = Path(
    "artifacts/market-observation/nfl/x15/regulation-decision-panel-v3"
)

FEATURE_BLOCKS: Final[Mapping[str, tuple[str, ...]]] = {
    "D0": ("landmark_seconds", "endpoint_seconds"),
    "D1": (
        "landmark_seconds",
        "endpoint_seconds",
        "mark_l_price",
        "mark_l_staleness_seconds",
        "prior_30s_actual_trade_count",
        "prior_30s_actual_trade_size",
        "prior_60s_actual_trade_count",
        "prior_60s_actual_trade_size",
    ),
    "D2": (
        "landmark_seconds",
        "endpoint_seconds",
        "mark_l_price",
        "mark_l_staleness_seconds",
        "prior_30s_actual_trade_count",
        "prior_30s_actual_trade_size",
        "prior_60s_actual_trade_count",
        "prior_60s_actual_trade_size",
        "quarter",
        "is_overtime",
        "game_seconds_remaining",
        "score_margin_home",
        "possession_is_home",
        "down",
        "distance",
        "yardline_100",
        "p_before_home",
        "p_before_home_missing",
    ),
    "D3": (
        "landmark_seconds",
        "endpoint_seconds",
        "mark_l_price",
        "mark_l_staleness_seconds",
        "prior_30s_actual_trade_count",
        "prior_30s_actual_trade_size",
        "prior_60s_actual_trade_count",
        "prior_60s_actual_trade_size",
        "quarter",
        "is_overtime",
        "game_seconds_remaining",
        "score_margin_home",
        "possession_is_home",
        "down",
        "distance",
        "yardline_100",
        "p_before_home",
        "p_before_home_missing",
        "action_group",
        "possession_result_group",
        "score_result_group",
        "kick_result_group",
        "adjudication_group",
        "yards_gained",
        "return_yards",
        "actor_is_home",
        "beneficiary_is_home",
    ),
}

_SHA_PREFIX = "sha256:"
_FINALIZED_SCHEMA = "nfl_x15_finalized_episode_batch_v1"
_CONTEXT_SCHEMA = "EventPrestateContextManifestV1"
_REQUIRED_SOURCE_HASH_KEYS = frozenset(
    {
        "finalized_manifest_sha256",
        "finalized_episode_object_sha256",
        "finalized_constituent_object_sha256",
        "context_manifest_sha256",
        "context_object_sha256",
        "facts_game_manifest_sha256",
        "facts_object_sha256",
        "market_game_manifest_sha256",
        "market_observations_object_sha256",
        "market_inventory_object_sha256",
        "cohort_authority_sha256",
    }
)


class RegulationPanelError(RuntimeError):
    """A source, time, target, feature, or publication invariant failed."""


@dataclass(frozen=True, slots=True)
class RegulationPanelSourceSpec:
    development: DevelopmentSourceSpec
    finalized_episode_manifest_path: Path = DEFAULT_FINALIZED_EPISODE_MANIFEST
    finalized_episode_manifest_sha256: str = (
        DEFAULT_FINALIZED_EPISODE_MANIFEST_SHA256
    )
    context_manifest_path: Path = DEFAULT_CONTEXT_MANIFEST
    context_manifest_sha256: str = DEFAULT_CONTEXT_MANIFEST_SHA256
    expected_game_count: int = EXPECTED_GAME_COUNT


@dataclass(frozen=True, slots=True)
class VerifiedRegulationPanelSources:
    development: VerifiedDevelopmentSources
    episodes: pd.DataFrame
    constituents: pd.DataFrame
    context: pd.DataFrame
    finalized_manifest: Mapping[str, Any]
    context_manifest: Mapping[str, Any]
    finalized_manifest_sha256: str
    context_manifest_sha256: str
    finalized_episode_object_sha256: str
    finalized_constituent_object_sha256: str
    context_object_sha256: str


@dataclass(frozen=True, slots=True)
class RegulationGamePanel:
    panel: pd.DataFrame
    attrition: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PublishedRegulationPanel:
    output_root: Path
    batch_manifest_path: Path
    batch_manifest_sha256: str
    batch_sha256: str
    game_count: int
    eligible_episode_count: int
    panel_row_count: int
    decision_eligible_count: int
    loss_eligible_count: int
    direction_eligible_count: int
    censored_count: int
    availability_zero_count: int
    availability_one_count: int
    runtime_seconds: float
    peak_rss_mib: float


@dataclass(frozen=True, slots=True)
class _TradeIndex:
    times_ns: np.ndarray
    prices: np.ndarray
    sizes: np.ndarray
    trade_ids: np.ndarray


@dataclass(frozen=True, slots=True)
class _Mark:
    status: str
    source_time: pd.Timestamp | None
    price: float
    trade_ids: tuple[str, ...]
    trade_id_set_sha256: str | None
    observation_count: int
    observed_size: float
    staleness_seconds: float

    @property
    def observed(self) -> bool:
        return self.status == "OBSERVED"


def default_source_spec() -> RegulationPanelSourceSpec:
    return RegulationPanelSourceSpec(development=default_development_source_spec())


def _sha256_bytes(payload: bytes) -> str:
    return _SHA_PREFIX + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return _SHA_PREFIX + digest.hexdigest()


def _require_sha(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith(_SHA_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise RegulationPanelError(f"{label} is not a canonical SHA-256")
    return value


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_canonical_value(child) for child in value]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    if hasattr(value, "item"):
        return _canonical_value(value.item())
    raise RegulationPanelError(
        f"unsupported canonical type {type(value).__name__}"
    )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _safe_path(root: Path, value: str | Path, *, label: str) -> Path:
    root = root.resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise RegulationPanelError(f"{label} escapes project root")
    return candidate


def _read_json_manifest(
    *,
    project_root: Path,
    path: Path,
    expected_sha256: str,
    expected_schema: str,
) -> tuple[dict[str, Any], Path]:
    expected = _require_sha(expected_sha256, label=f"{expected_schema}.sha256")
    resolved = _safe_path(project_root, path, label=expected_schema)
    if not resolved.is_file() or _sha256_file(resolved) != expected:
        raise RegulationPanelError(f"{expected_schema} manifest hash mismatch")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegulationPanelError(
            f"{expected_schema} manifest is unreadable"
        ) from exc
    if (
        type(document) is not dict
        or document.get("schema") != expected_schema
        or document.get("publication_gate") != "PASS"
        or document.get("holdout_reaction_accessed") is not False
    ):
        raise RegulationPanelError(f"{expected_schema} contract mismatch")
    material = dict(document)
    declared = material.pop("bundle_sha256", None)
    if declared != _canonical_sha(material):
        raise RegulationPanelError(f"{expected_schema} semantic hash mismatch")
    return document, resolved


def _manifest_table(
    *,
    project_root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    table_name: str,
) -> tuple[pd.DataFrame, str]:
    tables = manifest.get("tables")
    if type(tables) is not list:
        descriptor = manifest.get("context") if table_name == "context" else None
    else:
        matches = [
            row
            for row in tables
            if type(row) is dict and row.get("name") == table_name
        ]
        descriptor = matches[0] if len(matches) == 1 else None
    if type(descriptor) is not dict:
        raise RegulationPanelError(f"{table_name} descriptor missing")
    object_sha = _require_sha(
        descriptor.get("object_sha256"),
        label=f"{table_name}.object_sha256",
    )
    dataset_root = manifest_path.parents[3]
    relative = str(descriptor.get("object_path", ""))
    candidates = (dataset_root / relative, dataset_root.parent / relative)
    object_path = next(
        (
            candidate.resolve()
            for candidate in candidates
            if candidate.resolve().is_file()
        ),
        None,
    )
    if (
        object_path is None
        or project_root.resolve() not in object_path.parents
        or object_path.stat().st_size != descriptor.get("byte_length")
        or _sha256_file(object_path) != object_sha
    ):
        raise RegulationPanelError(f"{table_name} object verification failed")
    try:
        parquet = pq.ParquetFile(object_path)
        if parquet.metadata.num_rows != descriptor.get("row_count"):
            raise RegulationPanelError(f"{table_name} row-count mismatch")
        frame = parquet.read().to_pandas()
    except (OSError, pa.ArrowException) as exc:
        raise RegulationPanelError(f"{table_name} parquet unreadable") from exc
    if list(frame.columns) != descriptor.get("schema_columns"):
        raise RegulationPanelError(f"{table_name} schema-columns mismatch")
    return frame, object_sha


def verify_regulation_panel_sources(
    *,
    project_root: Path,
    source_spec: RegulationPanelSourceSpec | None = None,
) -> VerifiedRegulationPanelSources:
    """Verify all development sources without opening holdout reactions."""

    root = Path(project_root).resolve()
    spec = source_spec or default_source_spec()
    if spec.expected_game_count != EXPECTED_GAME_COUNT:
        raise RegulationPanelError("PanelV3 requires exactly 153 games")
    try:
        development = verify_development_sources(
            project_root=root,
            source_spec=spec.development,
        )
    except DevelopmentPanelError as exc:
        raise RegulationPanelError(str(exc)) from exc
    finalized, finalized_path = _read_json_manifest(
        project_root=root,
        path=spec.finalized_episode_manifest_path,
        expected_sha256=spec.finalized_episode_manifest_sha256,
        expected_schema=_FINALIZED_SCHEMA,
    )
    context_manifest, context_path = _read_json_manifest(
        project_root=root,
        path=spec.context_manifest_path,
        expected_sha256=spec.context_manifest_sha256,
        expected_schema=_CONTEXT_SCHEMA,
    )
    if (
        finalized.get("regulation_only") is not True
        or finalized.get("counts", {}).get("game_count")
        != spec.expected_game_count
        or finalized.get("counts", {}).get("stage_b_eligible_episode_count")
        != 21093
        or context_manifest.get("game_count") != spec.expected_game_count
        or context_manifest.get("market_data_read") is not False
    ):
        raise RegulationPanelError("episode/context cohort contract mismatch")
    episodes, episode_sha = _manifest_table(
        project_root=root,
        manifest_path=finalized_path,
        manifest=finalized,
        table_name="finalized_episodes",
    )
    constituents, constituent_sha = _manifest_table(
        project_root=root,
        manifest_path=finalized_path,
        manifest=finalized,
        table_name="episode_constituents",
    )
    context, context_sha = _manifest_table(
        project_root=root,
        manifest_path=context_path,
        manifest=context_manifest,
        table_name="context",
    )
    game_sets = (
        set(development.cohort_metadata["game_id"].astype(str)),
        set(episodes["game_id"].astype(str)),
        set(context["game_id"].astype(str)),
    )
    if not (game_sets[0] == game_sets[1] == game_sets[2]):
        raise RegulationPanelError("global source game sets differ")
    if (
        episodes.loc[episodes["stage_b_eligible"].eq(True), "quarter"]  # noqa: E712
        .astype("Int64")
        .gt(4)
        .any()
    ):
        raise RegulationPanelError("overtime leaked into finalized episodes")
    return VerifiedRegulationPanelSources(
        development=development,
        episodes=episodes,
        constituents=constituents,
        context=context,
        finalized_manifest=finalized,
        context_manifest=context_manifest,
        finalized_manifest_sha256=spec.finalized_episode_manifest_sha256,
        context_manifest_sha256=spec.context_manifest_sha256,
        finalized_episode_object_sha256=episode_sha,
        finalized_constituent_object_sha256=constituent_sha,
        context_object_sha256=context_sha,
    )


def _tags(value: object) -> set[str]:
    if value is None or value is pd.NA:
        return set()
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RegulationPanelError("outcome tags are invalid JSON") from exc
    elif isinstance(value, (list, tuple, set)):
        decoded = list(value)
    else:
        raise RegulationPanelError("outcome tags have unsupported type")
    if not isinstance(decoded, list) or any(
        not isinstance(item, str) for item in decoded
    ):
        raise RegulationPanelError("outcome tags must be a string list")
    return {item.strip().upper() for item in decoded if item.strip()}


def _finite_or_nan(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _nullable_bool(value: object) -> bool | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return bool(value)


def _trade_index(frame: pd.DataFrame) -> _TradeIndex:
    if frame.empty:
        return _TradeIndex(
            times_ns=np.array([], dtype=np.int64),
            prices=np.array([], dtype=float),
            sizes=np.array([], dtype=float),
            trade_ids=np.array([], dtype=object),
        )
    work = frame.copy()
    work["source_time_utc"] = pd.to_datetime(
        work["source_time_utc"], utc=True, errors="raise"
    )
    work["price"] = pd.to_numeric(work["price"], errors="raise")
    work["size"] = pd.to_numeric(work["size"], errors="raise")
    if (
        (~np.isfinite(work["price"])).any()
        or (~np.isfinite(work["size"])).any()
        or work["price"].lt(0).any()
        or work["price"].gt(1).any()
        or work["size"].le(0).any()
        or work["trade_id"].astype(str).duplicated().any()
    ):
        raise RegulationPanelError("market trades violate price/size/ID contract")
    work = work.sort_values(
        ["source_time_utc", "trade_id"], kind="mergesort"
    ).reset_index(drop=True)
    return _TradeIndex(
        # Pandas may preserve an Arrow-derived microsecond dtype.  Timestamp
        # ``.value`` is always nanoseconds and therefore matches the landmark
        # clock below.
        times_ns=np.fromiter(
            (
                pd.Timestamp(value).value
                for value in work["source_time_utc"]
            ),
            dtype=np.int64,
            count=len(work),
        ),
        prices=work["price"].astype(float).to_numpy(),
        sizes=work["size"].astype(float).to_numpy(),
        trade_ids=work["trade_id"].astype(str).to_numpy(dtype=object),
    )


def _bucket_mark(
    trades: _TradeIndex,
    *,
    lower_exclusive_ns: int | None,
    lower_inclusive_ns: int | None,
    upper_inclusive_ns: int,
    staleness_origin_ns: int,
) -> _Mark:
    if (lower_exclusive_ns is None) == (lower_inclusive_ns is None):
        raise AssertionError("exactly one lower bound must be supplied")
    lower = (
        int(np.searchsorted(trades.times_ns, lower_exclusive_ns, side="right"))
        if lower_exclusive_ns is not None
        else int(
            np.searchsorted(trades.times_ns, lower_inclusive_ns, side="left")
        )
    )
    upper = int(
        np.searchsorted(trades.times_ns, upper_inclusive_ns, side="right")
    )
    if lower >= upper:
        return _Mark(
            status="NO_ACTUAL_TRADE",
            source_time=None,
            price=math.nan,
            trade_ids=(),
            trade_id_set_sha256=None,
            observation_count=0,
            observed_size=0.0,
            staleness_seconds=math.nan,
        )
    latest_ns = int(trades.times_ns[upper - 1])
    bucket_start = max(
        lower,
        int(np.searchsorted(trades.times_ns, latest_ns, side="left")),
    )
    bucket_end = int(
        np.searchsorted(trades.times_ns, latest_ns, side="right")
    )
    prices = trades.prices[bucket_start:bucket_end]
    sizes = trades.sizes[bucket_start:bucket_end]
    ids = tuple(sorted(map(str, trades.trade_ids[bucket_start:bucket_end])))
    total_size = float(sizes.sum())
    if (
        not ids
        or not np.isfinite(prices).all()
        or not np.isfinite(sizes).all()
        or np.any(sizes <= 0)
        or total_size <= 0
    ):
        raise RegulationPanelError("same-time actual-trade bucket is invalid")
    return _Mark(
        status="OBSERVED",
        source_time=pd.Timestamp(latest_ns, tz="UTC"),
        price=float(np.dot(prices, sizes) / total_size),
        trade_ids=ids,
        trade_id_set_sha256=_canonical_sha(list(ids)),
        observation_count=len(ids),
        observed_size=total_size,
        staleness_seconds=float(
            (staleness_origin_ns - latest_ns) / 1_000_000_000
        ),
    )


def _activity(
    trades: _TradeIndex,
    *,
    decision_ns: int,
    seconds: int,
) -> tuple[int, float]:
    lower = decision_ns - seconds * 1_000_000_000
    first = int(np.searchsorted(trades.times_ns, lower, side="right"))
    after = int(np.searchsorted(trades.times_ns, decision_ns, side="right"))
    return after - first, float(trades.sizes[first:after].sum())


def _axis_groups(
    *,
    episode: pd.Series,
    fact: pd.Series,
) -> dict[str, object]:
    tags = _tags(episode["final_outcome_tags"])
    action = str(fact.get("primary_action", "UNKNOWN")).upper()
    action_group = (
        action
        if action
        in {
            "PASS",
            "RUN",
            "SACK",
            "PUNT",
            "KICKOFF",
            "FIELD_GOAL",
            "TRY",
            "KNEEL",
        }
        else "OTHER"
    )
    if "INTERCEPTION" in tags:
        possession = "INTERCEPTION"
    elif "LOST_FUMBLE" in tags:
        possession = "LOST_FUMBLE"
    elif "TURNOVER_ON_DOWNS" in tags:
        possession = "TURNOVER_ON_DOWNS"
    elif "MUFFED_PUNT" in tags:
        possession = "MUFFED_PUNT"
    elif "MUFFED_KICKOFF" in tags:
        possession = "MUFFED_KICKOFF"
    elif "POSSESSION_CHANGE" in tags:
        possession = "OTHER_POSSESSION_CHANGE"
    else:
        possession = "POSSESSION_RETAINED"

    if "PASS_TOUCHDOWN" in tags:
        score = "PASS_TOUCHDOWN"
    elif "RUSH_TOUCHDOWN" in tags:
        score = "RUSH_TOUCHDOWN"
    elif "RETURN_TOUCHDOWN" in tags:
        score = "RETURN_TOUCHDOWN"
    elif "DEFENSIVE_TWO_POINT" in tags:
        score = "DEFENSIVE_TWO_POINT"
    elif "SAFETY" in tags:
        score = "SAFETY"
    elif "FIELD_GOAL_MADE" in tags:
        score = "FIELD_GOAL"
    elif "EXTRA_POINT_GOOD" in tags:
        score = "EXTRA_POINT"
    elif "TWO_POINT_SUCCESS" in tags:
        score = "TWO_POINT"
    elif int(episode.get("score_points", 0) or 0) > 0:
        score = "OTHER_SCORE"
    else:
        score = "NO_SCORE"

    kick_related = action in {"PUNT", "KICKOFF", "FIELD_GOAL", "TRY"}
    if "BLOCKED_KICK" in tags or any(tag.startswith("BLOCKED_") for tag in tags):
        kick = "BLOCKED"
    elif "ONSIDE_RECOVERY" in tags or "ONSIDE_KICK_RECOVERED" in tags:
        kick = "ONSIDE_RECOVERED"
    elif "KICKOFF_TOUCHBACK" in tags or "PUNT_TOUCHBACK" in tags:
        kick = "TOUCHBACK"
    elif "PUNT_FAIR_CATCH" in tags:
        kick = "FAIR_CATCH"
    elif "PUNT_INSIDE_20" in tags:
        kick = "INSIDE_20"
    elif "RETURN_TOUCHDOWN" in tags and kick_related:
        kick = "RETURN_TOUCHDOWN"
    elif any("RETURN" in tag for tag in tags) and kick_related:
        kick = "RETURN"
    elif "FIELD_GOAL_MADE" in tags or "EXTRA_POINT_GOOD" in tags:
        kick = "MADE"
    elif (
        "FIELD_GOAL_MISSED" in tags
        or "EXTRA_POINT_MISSED" in tags
        or "TWO_POINT_FAILED" in tags
    ):
        kick = "MISSED_OR_FAILED"
    elif kick_related:
        kick = "OTHER_KICK"
    else:
        kick = "NO_KICK"

    review = str(fact.get("review_result", "") or "").strip().upper()
    if "REVERSAL" in tags or review == "REVERSED":
        adjudication = "REVIEW_REVERSED"
    elif "REVIEW" in tags and review:
        adjudication = f"REVIEW_{review}"
    elif "REVIEW" in tags:
        adjudication = "REVIEW_UNKNOWN_RESULT"
    elif "PENALTY_ACCEPTED_NO_PLAY" in tags:
        adjudication = "PENALTY_ACCEPTED_NO_PLAY"
    elif "PENALTY_ACCEPTED" in tags:
        adjudication = "PENALTY_ACCEPTED"
    elif "PENALTY_OFFSETTING" in tags:
        adjudication = "PENALTY_OFFSETTING"
    elif "PENALTY_DECLINED" in tags:
        adjudication = "PENALTY_DECLINED"
    elif "PENALTY" in tags:
        adjudication = "PENALTY_OTHER"
    else:
        adjudication = "NO_ADJUDICATION"
    return {
        "action_group": action_group,
        "possession_result_group": possession,
        "score_result_group": score,
        "kick_result_group": kick,
        "adjudication_group": adjudication,
    }


def _direction(delta: float) -> str:
    if delta >= DIRECTION_THRESHOLD:
        return "UP"
    if delta <= -DIRECTION_THRESHOLD:
        return "DOWN"
    return "NO_MOVE"


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RegulationPanelError(f"{label} missing columns: {missing}")


def build_regulation_game_panel(
    *,
    episodes: pd.DataFrame,
    constituents: pd.DataFrame,
    context: pd.DataFrame,
    facts: pd.DataFrame,
    market_rows: pd.DataFrame,
    contracts: pd.DataFrame,
    cohort_row: Mapping[str, object],
    source_hashes: Mapping[str, str],
    landmark_seconds: Sequence[int] = LANDMARK_SECONDS,
    endpoint_seconds: Sequence[int] = ENDPOINT_SECONDS,
) -> RegulationGamePanel:
    """Build one game's full PanelV3 rows without publishing them."""

    _require_columns(
        episodes,
        {
            "game_id",
            "episode_id",
            "episode_status",
            "episode_type",
            "constituent_count",
            "primary_event_id",
            "quarter",
            "first_seen_interval_start",
            "first_seen_interval_end",
            "final_known_interval_start",
            "final_known_interval_end",
            "episode_anchor",
            "stage_b_eligible",
            "home_team",
            "away_team",
            "beneficiary_team",
            "pre_home_score",
            "pre_away_score",
            "post_home_score",
            "post_away_score",
            "score_points",
            "turnover",
            "final_outcome_tags",
            "source_hashes",
        },
        label="episodes",
    )
    _require_columns(
        constituents,
        {"game_id", "event_id", "episode_id", "mapping_status", "order_sequence"},
        label="constituents",
    )
    _require_columns(
        context,
        {
            "game_id",
            "event_id",
            "is_regulation",
            "pre_state_known_at",
            "p_before_home",
            "prestate_support_status",
            "source_binding_sha256",
        },
        label="context",
    )
    _require_columns(
        facts,
        {
            "game_id",
            "event_id",
            "quarter",
            "game_clock",
            "game_seconds_remaining",
            "home_team",
            "away_team",
            "actor_is_home",
            "beneficiary_is_home",
            "possession_is_home",
            "pre_home_score",
            "pre_away_score",
            "score_margin_home",
            "down",
            "distance",
            "yardline_100",
            "primary_action",
            "yards_gained",
            "return_yards",
            "review_result",
            "outcome_tags",
        },
        label="facts",
    )
    missing_hashes = _REQUIRED_SOURCE_HASH_KEYS - set(source_hashes)
    if missing_hashes:
        raise RegulationPanelError(
            f"source hashes missing keys: {sorted(missing_hashes)}"
        )
    normalized_hashes = {
        key: _require_sha(value, label=f"source_hashes.{key}")
        for key, value in sorted(source_hashes.items())
    }
    game_ids = {
        *episodes["game_id"].astype(str).unique(),
        *constituents["game_id"].astype(str).unique(),
        *context["game_id"].astype(str).unique(),
        *facts["game_id"].astype(str).unique(),
    }
    if len(game_ids) != 1:
        raise RegulationPanelError("single-game inputs crossed game boundary")
    game_id = next(iter(game_ids))
    if str(cohort_row["game_id"]) != game_id:
        raise RegulationPanelError("cohort row game mismatch")
    if episodes["episode_id"].astype(str).duplicated().any():
        raise RegulationPanelError("episode identity is not unique")
    if facts["event_id"].astype(str).duplicated().any():
        raise RegulationPanelError("fact event identity is not unique")
    if context["event_id"].astype(str).duplicated().any():
        raise RegulationPanelError("context event identity is not unique")
    if constituents.loc[
        constituents["mapping_status"].eq("MAPPED"), "event_id"
    ].astype(str).duplicated().any():
        raise RegulationPanelError("constituent event maps more than once")

    l_values = tuple(sorted(set(map(int, landmark_seconds))))
    h_values = tuple(sorted(set(map(int, endpoint_seconds))))
    if (
        not l_values
        or not h_values
        or any(value <= 0 for value in (*l_values, *h_values))
    ):
        raise RegulationPanelError("time grid must contain positive seconds")

    episode_work = episodes.copy()
    for column in (
        "first_seen_interval_start",
        "first_seen_interval_end",
        "final_known_interval_start",
        "final_known_interval_end",
        "episode_anchor",
    ):
        episode_work[column] = pd.to_datetime(
            episode_work[column],
            utc=True,
            errors="coerce",
            format="mixed",
        )
    mapped = constituents.loc[
        constituents["mapping_status"].eq("MAPPED")
        & constituents["episode_id"].notna()
    ].copy()
    orders = (
        mapped.groupby("episode_id", sort=False)["order_sequence"]
        .min()
        .rename("_episode_order")
    )
    episode_work = episode_work.join(orders, on="episode_id")
    if episode_work["_episode_order"].isna().any():
        raise RegulationPanelError("episode lacks mapped constituent order")
    eligible = episode_work.loc[
        episode_work["stage_b_eligible"].eq(True)  # noqa: E712
    ].copy()
    if (
        eligible.empty
        or eligible["episode_status"].ne("FINAL").any()
        or eligible["episode_anchor"].isna().any()
        or eligible["quarter"].astype("Int64").gt(4).any()
    ):
        raise RegulationPanelError("Stage-B focal episodes are not regulation FINAL")

    timeline = episode_work.loc[
        episode_work["episode_anchor"].notna(),
        ["episode_id", "episode_anchor", "_episode_order"],
    ].sort_values(
        ["episode_anchor", "_episode_order", "episode_id"], kind="mergesort"
    )
    next_by_episode: dict[str, pd.Timestamp | None] = {}
    timeline_records = timeline.to_dict("records")
    for index, record in enumerate(timeline_records):
        next_by_episode[str(record["episode_id"])] = (
            timeline_records[index + 1]["episode_anchor"]
            if index + 1 < len(timeline_records)
            else None
        )

    facts_index = {
        str(row["event_id"]): pd.Series(row) for row in facts.to_dict("records")
    }
    context_index = {
        str(row["event_id"]): pd.Series(row)
        for row in context.to_dict("records")
    }
    home_contracts = contracts.loc[
        contracts["contract_role"].astype(str).eq("ACTUAL_HOME_OUTCOME")
    ].copy()
    if (
        home_contracts.empty
        or home_contracts.duplicated(["venue", "contract_id"]).any()
    ):
        raise RegulationPanelError("actual home contract identity is invalid")
    trade_indices = {
        (str(venue), str(contract_id)): _trade_index(group)
        for (venue, contract_id), group in market_rows.groupby(
            ["venue", "contract_id"], sort=False
        )
    }
    empty_index = _trade_index(market_rows.iloc[0:0])
    source_hashes_json = json.dumps(
        normalized_hashes, sort_keys=True, separators=(",", ":")
    )
    source_binding_sha256 = _canonical_sha(normalized_hashes)

    rows: list[dict[str, object]] = []
    for raw_episode in eligible.sort_values(
        ["episode_anchor", "_episode_order", "episode_id"], kind="mergesort"
    ).to_dict("records"):
        episode = pd.Series(raw_episode)
        event_id = str(episode["primary_event_id"])
        fact = facts_index.get(event_id)
        prestate = context_index.get(event_id)
        if fact is None or prestate is None:
            raise RegulationPanelError(
                f"{game_id}/{event_id} lacks fact or pre-state context"
            )
        if (
            not bool(prestate["is_regulation"])
            or int(fact["quarter"]) > 4
            or int(episode["quarter"]) > 4
        ):
            raise RegulationPanelError("overtime leaked into PanelV3")
        anchor = pd.Timestamp(episode["episode_anchor"])
        prestate_known = pd.to_datetime(
            prestate["pre_state_known_at"],
            utc=True,
            errors="coerce",
        )
        if pd.notna(prestate_known) and prestate_known > anchor:
            raise RegulationPanelError("pre-state was not known by episode anchor")
        supported = str(prestate["prestate_support_status"]) == "SUPPORTED"
        p_before = _finite_or_nan(prestate["p_before_home"])
        if supported and (not math.isfinite(p_before) or not 0 <= p_before <= 1):
            raise RegulationPanelError("supported p_before_home is invalid")
        if not supported:
            p_before = math.nan
        axes = _axis_groups(episode=episode, fact=fact)
        next_anchor = next_by_episode.get(str(episode["episode_id"]))

        for contract in home_contracts.to_dict("records"):
            venue = str(contract["venue"])
            contract_id = str(contract["contract_id"])
            trades = trade_indices.get((venue, contract_id), empty_index)
            anchor_ns = anchor.value
            for l_seconds in l_values:
                landmark = anchor + pd.Timedelta(seconds=l_seconds)
                landmark_ns = landmark.value
                mark_l = _bucket_mark(
                    trades,
                    lower_exclusive_ns=None,
                    lower_inclusive_ns=anchor_ns,
                    upper_inclusive_ns=landmark_ns,
                    staleness_origin_ns=landmark_ns,
                )
                count_30, size_30 = _activity(
                    trades, decision_ns=landmark_ns, seconds=30
                )
                count_60, size_60 = _activity(
                    trades, decision_ns=landmark_ns, seconds=60
                )
                censor_at_l = (
                    next_anchor is None or pd.Timestamp(next_anchor) <= landmark
                )
                decision_eligible = bool(mark_l.observed and not censor_at_l)
                for h_seconds in h_values:
                    if h_seconds <= l_seconds:
                        continue
                    endpoint = anchor + pd.Timedelta(seconds=h_seconds)
                    endpoint_ns = endpoint.value
                    if next_anchor is None:
                        censored = True
                        censor_reason = "CONTINUITY_BOUND_UNKNOWN_AFTER_FINAL_EPISODE"
                        censor_time = pd.NaT
                    elif pd.Timestamp(next_anchor) <= endpoint:
                        censored = True
                        censor_time = pd.Timestamp(next_anchor)
                        censor_reason = (
                            "NEXT_FINALIZED_EPISODE_AT_OR_BEFORE_L"
                            if pd.Timestamp(next_anchor) <= landmark
                            else "NEXT_FINALIZED_EPISODE_IN_L_H_WINDOW"
                        )
                    else:
                        censored = False
                        censor_time = pd.Timestamp(next_anchor)
                        censor_reason = "NOT_CENSORED"
                    mark_h = _bucket_mark(
                        trades,
                        lower_exclusive_ns=landmark_ns,
                        lower_inclusive_ns=None,
                        upper_inclusive_ns=endpoint_ns,
                        staleness_origin_ns=endpoint_ns,
                    )
                    loss_eligible = bool(decision_eligible and not censored)
                    availability: int | None = (
                        int(mark_h.observed) if loss_eligible else None
                    )
                    direction: str | None = None
                    delta = math.nan
                    if loss_eligible and mark_h.observed:
                        delta = mark_h.price - mark_l.price
                        direction = _direction(delta)
                    direction_eligible = direction is not None
                    if censored:
                        attrition_reason = censor_reason
                    elif not mark_l.observed:
                        attrition_reason = "LANDMARK_NO_ACTUAL_TRADE"
                    elif mark_h.observed:
                        attrition_reason = "ELIGIBLE_AVAILABILITY_ONE"
                    else:
                        attrition_reason = "ELIGIBLE_AVAILABILITY_ZERO"

                    identity = {
                        "schema_version": SCHEMA_VERSION,
                        "game_id": game_id,
                        "finalized_episode_id": str(episode["episode_id"]),
                        "venue": venue,
                        "actual_home_contract_id": contract_id,
                        "landmark_seconds": l_seconds,
                        "endpoint_seconds": h_seconds,
                        "source_binding_sha256": source_binding_sha256,
                    }
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "claim_boundary": CLAIM_BOUNDARY,
                            "experiment_id": EXPERIMENT_ID,
                            "dataset_split": DATASET_SPLIT,
                            "holdout_reaction_accessed": False,
                            "cohort_authority_sha256": str(
                                cohort_row["authority_sha256"]
                            ),
                            "source_row_id": _canonical_sha(identity),
                            "source_binding_sha256": source_binding_sha256,
                            "source_hashes_json": source_hashes_json,
                            "game_id": game_id,
                            "nfl_week": int(cohort_row["nfl_week"]),
                            "finalized_episode_id": str(episode["episode_id"]),
                            "primary_event_id": event_id,
                            "venue": venue,
                            "actual_home_contract_id": contract_id,
                            "home_team": str(episode["home_team"]),
                            "away_team": str(episode["away_team"]),
                            "target_orientation": "ACTUAL_HOME_OUTCOME",
                            "episode_anchor": anchor,
                            "episode_anchor_semantics": (
                                "END_OF_LAST_CONSTITUENT_DECIDING_FINAL_SEMANTICS"
                            ),
                            "next_finalized_episode_anchor": next_anchor,
                            "censor_time_utc": censor_time,
                            "landmark_seconds": l_seconds,
                            "endpoint_seconds": h_seconds,
                            "landmark_utc": landmark,
                            "endpoint_utc": endpoint,
                            "decision_eligible": decision_eligible,
                            "censored": censored,
                            "censor_reason": censor_reason,
                            "loss_eligible": loss_eligible,
                            "availability_h": availability,
                            "direction_h": direction,
                            "direction_eligible": direction_eligible,
                            "delta_l_h": delta,
                            "direction_threshold_probability": (
                                DIRECTION_THRESHOLD
                            ),
                            "mark_l_status": mark_l.status,
                            "mark_l_source_time_utc": mark_l.source_time,
                            "mark_l_price": (
                                mark_l.price if mark_l.observed else math.nan
                            ),
                            "mark_l_staleness_seconds": (
                                mark_l.staleness_seconds
                            ),
                            "mark_l_trade_ids_json": json.dumps(
                                list(mark_l.trade_ids), separators=(",", ":")
                            ),
                            "mark_l_trade_id_set_sha256": (
                                mark_l.trade_id_set_sha256
                            ),
                            "mark_l_observation_count": (
                                mark_l.observation_count
                            ),
                            "mark_l_observed_size": mark_l.observed_size,
                            "mark_h_status": mark_h.status,
                            "mark_h_source_time_utc": mark_h.source_time,
                            "mark_h_price": (
                                mark_h.price if mark_h.observed else math.nan
                            ),
                            "mark_h_staleness_seconds": (
                                mark_h.staleness_seconds
                            ),
                            "mark_h_trade_ids_json": json.dumps(
                                list(mark_h.trade_ids), separators=(",", ":")
                            ),
                            "mark_h_trade_id_set_sha256": (
                                mark_h.trade_id_set_sha256
                            ),
                            "mark_h_observation_count": (
                                mark_h.observation_count
                            ),
                            "mark_h_observed_size": mark_h.observed_size,
                            "prior_30s_actual_trade_count": count_30,
                            "prior_30s_actual_trade_size": size_30,
                            "prior_60s_actual_trade_count": count_60,
                            "prior_60s_actual_trade_size": size_60,
                            "quarter": int(fact["quarter"]),
                            "is_overtime": False,
                            "game_clock": fact["game_clock"],
                            "game_seconds_remaining": _finite_or_nan(
                                fact["game_seconds_remaining"]
                            ),
                            "pre_home_score": _finite_or_nan(
                                fact["pre_home_score"]
                            ),
                            "pre_away_score": _finite_or_nan(
                                fact["pre_away_score"]
                            ),
                            "score_margin_home": _finite_or_nan(
                                fact["score_margin_home"]
                            ),
                            "possession_is_home": _nullable_bool(
                                fact["possession_is_home"]
                            ),
                            "down": _finite_or_nan(fact["down"]),
                            "distance": _finite_or_nan(fact["distance"]),
                            "yardline_100": _finite_or_nan(
                                fact["yardline_100"]
                            ),
                            "p_before_home": p_before,
                            "p_before_home_missing": not supported,
                            "prestate_support_status": str(
                                prestate["prestate_support_status"]
                            ),
                            "pre_state_known_at": prestate_known,
                            "prestate_source_binding_sha256": str(
                                prestate["source_binding_sha256"]
                            ),
                            **axes,
                            "yards_gained": _finite_or_nan(
                                fact["yards_gained"]
                            ),
                            "return_yards": _finite_or_nan(
                                fact["return_yards"]
                            ),
                            "actor_is_home": _nullable_bool(
                                fact["actor_is_home"]
                            ),
                            "beneficiary_is_home": _nullable_bool(
                                fact["beneficiary_is_home"]
                            ),
                            "episode_score_points": int(
                                episode["score_points"] or 0
                            ),
                            "episode_turnover": bool(episode["turnover"]),
                            "episode_constituent_count": int(
                                episode["constituent_count"]
                            ),
                            "attrition_reason": attrition_reason,
                        }
                    )
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RegulationPanelError("single-game PanelV3 unexpectedly empty")
    panel["availability_h"] = pd.array(
        panel["availability_h"], dtype="Int8"
    )
    panel["direction_h"] = panel["direction_h"].astype("string")
    grain = [
        "game_id",
        "finalized_episode_id",
        "venue",
        "actual_home_contract_id",
        "landmark_seconds",
        "endpoint_seconds",
    ]
    if panel.duplicated(grain).any() or panel["source_row_id"].duplicated().any():
        raise RegulationPanelError("PanelV3 grain/source_row_id is not unique")
    if (
        panel["is_overtime"].any()
        or panel.loc[panel["censored"], "loss_eligible"].any()
        or panel.loc[
            panel["censored"], "availability_h"
        ].notna().any()
        or panel.filter(regex=r"^(factor__|event_tag__)").shape[1]
    ):
        raise RegulationPanelError("PanelV3 scope/target/feature invariant failed")
    expected_rows = (
        eligible["episode_id"].nunique()
        * len(home_contracts)
        * sum(1 for l in l_values for h in h_values if h > l)
    )
    if len(panel) != expected_rows:
        raise RegulationPanelError("PanelV3 row reconciliation failed")
    panel = panel.sort_values(grain, kind="mergesort").reset_index(drop=True)
    attrition = (
        panel.groupby(
            ["game_id", "venue", "landmark_seconds", "endpoint_seconds"],
            sort=True,
            dropna=False,
        )
        .agg(
            panel_rows=("source_row_id", "size"),
            decision_eligible_rows=("decision_eligible", "sum"),
            censored_rows=("censored", "sum"),
            loss_eligible_rows=("loss_eligible", "sum"),
            direction_eligible_rows=("direction_eligible", "sum"),
            availability_zero_rows=(
                "availability_h",
                lambda values: int(values.eq(0).sum()),
            ),
            availability_one_rows=(
                "availability_h",
                lambda values: int(values.eq(1).sum()),
            ),
        )
        .reset_index()
    )
    attrition.insert(0, "schema_version", ATTRITION_SCHEMA)
    attrition["holdout_reaction_accessed"] = False
    return RegulationGamePanel(panel=panel, attrition=attrition)


def _parquet_bytes(frame: pd.DataFrame) -> tuple[bytes, str]:
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    return (
        sink.getvalue().to_pybytes(),
        _sha256_bytes(table.schema.remove_metadata().serialize().to_pybytes()),
    )


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RegulationPanelError(f"content-addressed collision: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != payload:
        raise RegulationPanelError(f"atomic publication failed: {path}")


def _publish_frame(
    *,
    output_root: Path,
    game_id: str,
    name: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    payload, schema_sha = _parquet_bytes(frame)
    object_sha = _sha256_bytes(payload)
    hexadecimal = object_sha.removeprefix(_SHA_PREFIX)
    relative = (
        Path("single-game")
        / game_id
        / "objects"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.parquet"
    )
    path = _safe_path(output_root, relative, label=f"{game_id}.{name}")
    _atomic_publish(path, payload)
    return {
        "name": name,
        "object_path": relative.as_posix(),
        "object_sha256": object_sha,
        "byte_length": len(payload),
        "row_count": len(frame),
        "schema_columns": list(frame.columns),
        "schema_fingerprint": schema_sha,
    }


def _peak_rss_mib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024


def publish_exact153_regulation_panel(
    *,
    project_root: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_spec: RegulationPanelSourceSpec | None = None,
) -> PublishedRegulationPanel:
    """Stream all 153 games through the corrected panel and publish atomically."""

    started = time.perf_counter()
    project = Path(project_root).resolve()
    output = _safe_path(project, output_root, label="output_root")
    sources = verify_regulation_panel_sources(
        project_root=project,
        source_spec=source_spec,
    )
    game_entries: list[dict[str, object]] = []
    aggregate = {
        "eligible_episode_count": 0,
        "panel_row_count": 0,
        "decision_eligible_count": 0,
        "loss_eligible_count": 0,
        "direction_eligible_count": 0,
        "censored_count": 0,
        "availability_zero_count": 0,
        "availability_one_count": 0,
    }
    for game_id in sorted(sources.development.facts.games):
        game_started = time.perf_counter()
        try:
            facts_manifest, _ = _verify_game_manifest(
                batch=sources.development.facts,
                descriptor=sources.development.facts.games[game_id],
                label="Facts V4",
            )
            market_manifest, _ = _verify_game_manifest(
                batch=sources.development.market,
                descriptor=sources.development.market.games[game_id],
                label="market",
            )
            facts = _read_verified_table(
                batch=sources.development.facts,
                manifest=facts_manifest,
                table_name="canonical_factor_events",
                market_style=False,
                label=f"Facts V4.{game_id}",
            )
            observations = _read_verified_table(
                batch=sources.development.market,
                manifest=market_manifest,
                table_name="actual_market_observations",
                market_style=True,
                label=f"market.{game_id}",
            )
            inventory = _read_verified_table(
                batch=sources.development.market,
                manifest=market_manifest,
                table_name="contract_inventory",
                market_style=True,
                label=f"market.{game_id}",
            )
        except DevelopmentPanelError as exc:
            raise RegulationPanelError(str(exc)) from exc
        teams = facts.loc[:, ["home_team", "away_team"]].drop_duplicates()
        if len(teams) != 1:
            raise RegulationPanelError(f"{game_id} team identity inconsistent")
        market_rows, contracts, _ = _adapt_market(
            game_id=game_id,
            home_team=str(teams.iloc[0]["home_team"]),
            observations=observations,
            inventory=inventory,
        )
        facts_descriptor = _table_descriptor(
            facts_manifest,
            table_name="canonical_factor_events",
            market_style=False,
            label=f"Facts V4.{game_id}",
        )
        observation_descriptor = _table_descriptor(
            market_manifest,
            table_name="actual_market_observations",
            market_style=True,
            label=f"market.{game_id}",
        )
        inventory_descriptor = _table_descriptor(
            market_manifest,
            table_name="contract_inventory",
            market_style=True,
            label=f"market.{game_id}",
        )
        source_hashes = {
            "finalized_manifest_sha256": (
                sources.finalized_manifest_sha256
            ),
            "finalized_episode_object_sha256": (
                sources.finalized_episode_object_sha256
            ),
            "finalized_constituent_object_sha256": (
                sources.finalized_constituent_object_sha256
            ),
            "context_manifest_sha256": sources.context_manifest_sha256,
            "context_object_sha256": sources.context_object_sha256,
            "facts_game_manifest_sha256": str(
                sources.development.facts.games[game_id]["manifest_sha256"]
            ),
            "facts_object_sha256": str(
                facts_descriptor["object_sha256"]
            ),
            "market_game_manifest_sha256": str(
                sources.development.market.games[game_id]["manifest_sha256"]
            ),
            "market_observations_object_sha256": str(
                observation_descriptor["object_sha256"]
            ),
            "market_inventory_object_sha256": str(
                inventory_descriptor["object_sha256"]
            ),
            "cohort_authority_sha256": (
                sources.development.cohort_authority_sha256
            ),
        }
        cohort = sources.development.cohort_metadata.loc[
            sources.development.cohort_metadata["game_id"].eq(game_id)
        ]
        built = build_regulation_game_panel(
            episodes=sources.episodes.loc[
                sources.episodes["game_id"].astype(str).eq(game_id)
            ].reset_index(drop=True),
            constituents=sources.constituents.loc[
                sources.constituents["game_id"].astype(str).eq(game_id)
            ].reset_index(drop=True),
            context=sources.context.loc[
                sources.context["game_id"].astype(str).eq(game_id)
            ].reset_index(drop=True),
            facts=facts,
            market_rows=market_rows,
            contracts=contracts,
            cohort_row=cohort.iloc[0].to_dict(),
            source_hashes=source_hashes,
        )
        descriptors = [
            _publish_frame(
                output_root=output,
                game_id=game_id,
                name="regulation_decision_panel",
                frame=built.panel,
            ),
            _publish_frame(
                output_root=output,
                game_id=game_id,
                name="attrition",
                frame=built.attrition,
            ),
        ]
        counts = {
            "eligible_episode_count": int(
                built.panel["finalized_episode_id"].nunique()
            ),
            "panel_row_count": len(built.panel),
            "decision_eligible_count": int(
                built.panel["decision_eligible"].sum()
            ),
            "loss_eligible_count": int(built.panel["loss_eligible"].sum()),
            "direction_eligible_count": int(
                built.panel["direction_eligible"].sum()
            ),
            "censored_count": int(built.panel["censored"].sum()),
            "availability_zero_count": int(
                built.panel["availability_h"].eq(0).sum()
            ),
            "availability_one_count": int(
                built.panel["availability_h"].eq(1).sum()
            ),
        }
        for key, value in counts.items():
            aggregate[key] += value
        game_material = {
            "schema": GAME_MANIFEST_SCHEMA,
            "builder_version": BUILDER_VERSION,
            "claim_boundary": CLAIM_BOUNDARY,
            "experiment_id": EXPERIMENT_ID,
            "dataset_split": DATASET_SPLIT,
            "game_id": game_id,
            "holdout_reaction_accessed": False,
            "regulation_only": True,
            "source_hashes": source_hashes,
            "feature_blocks": {
                key: list(value) for key, value in FEATURE_BLOCKS.items()
            },
            "counts": counts,
            "tables": descriptors,
            "publication_gate": "PASS",
        }
        game_bundle = _canonical_sha(game_material)
        game_manifest = {**game_material, "bundle_sha256": game_bundle}
        payload = _canonical_bytes(game_manifest)
        manifest_sha = _sha256_bytes(payload)
        hexadecimal = manifest_sha.removeprefix(_SHA_PREFIX)
        relative = (
            Path("single-game")
            / game_id
            / "manifests"
            / "sha256"
            / hexadecimal[:2]
            / f"{hexadecimal}.manifest.json"
        )
        manifest_path = _safe_path(
            output, relative, label=f"{game_id}.manifest"
        )
        _atomic_publish(manifest_path, payload)
        game_entries.append(
            {
                "game_id": game_id,
                "manifest_path": relative.as_posix(),
                "manifest_sha256": manifest_sha,
                "bundle_sha256": game_bundle,
                "panel_object_sha256": descriptors[0]["object_sha256"],
                "panel_row_count": len(built.panel),
                "runtime_seconds_observed": round(
                    time.perf_counter() - game_started, 6
                ),
            }
        )
        del facts, observations, inventory, market_rows, contracts, built

    if (
        len(game_entries) != EXPECTED_GAME_COUNT
        or aggregate["eligible_episode_count"] != 21093
    ):
        raise RegulationPanelError("exact-153 publication reconciliation failed")
    # Runtime is deliberately excluded from the semantic batch identity.
    deterministic_entries = [
        {
            key: value
            for key, value in entry.items()
            if key != "runtime_seconds_observed"
        }
        for entry in game_entries
    ]
    batch_material = {
        "schema": BATCH_MANIFEST_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "experiment_id": EXPERIMENT_ID,
        "dataset_split": DATASET_SPLIT,
        "holdout_reaction_accessed": False,
        "regulation_only": True,
        "game_count": len(game_entries),
        "time_grid": {
            "landmark_seconds": list(LANDMARK_SECONDS),
            "endpoint_seconds": list(ENDPOINT_SECONDS),
            "require_h_gt_l": True,
        },
        "direction_threshold_probability": DIRECTION_THRESHOLD,
        "feature_blocks": {
            key: list(value) for key, value in FEATURE_BLOCKS.items()
        },
        "sources": {
            "facts_batch_file_sha256": (
                sources.development.spec.facts_batch_file_sha256
            ),
            "market_batch_file_sha256": (
                sources.development.spec.market_batch_file_sha256
            ),
            "finalized_manifest_sha256": (
                sources.finalized_manifest_sha256
            ),
            "context_manifest_sha256": sources.context_manifest_sha256,
            "cohort_authority_sha256": (
                sources.development.cohort_authority_sha256
            ),
            "cohort_mapping_sha256": (
                sources.development.cohort_mapping_sha256
            ),
        },
        "counts": aggregate,
        "games": deterministic_entries,
        "publication_gate": "PASS",
    }
    batch_sha = _canonical_sha(batch_material)
    batch_manifest = {**batch_material, "batch_sha256": batch_sha}
    batch_payload = _canonical_bytes(batch_manifest)
    batch_manifest_sha = _sha256_bytes(batch_payload)
    hexadecimal = batch_manifest_sha.removeprefix(_SHA_PREFIX)
    relative = (
        Path("batches")
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.batch-index.json"
    )
    batch_path = _safe_path(output, relative, label="batch_manifest")
    _atomic_publish(batch_path, batch_payload)
    return PublishedRegulationPanel(
        output_root=output,
        batch_manifest_path=batch_path,
        batch_manifest_sha256=batch_manifest_sha,
        batch_sha256=batch_sha,
        game_count=len(game_entries),
        eligible_episode_count=aggregate["eligible_episode_count"],
        panel_row_count=aggregate["panel_row_count"],
        decision_eligible_count=aggregate["decision_eligible_count"],
        loss_eligible_count=aggregate["loss_eligible_count"],
        direction_eligible_count=aggregate["direction_eligible_count"],
        censored_count=aggregate["censored_count"],
        availability_zero_count=aggregate["availability_zero_count"],
        availability_one_count=aggregate["availability_one_count"],
        runtime_seconds=time.perf_counter() - started,
        peak_rss_mib=_peak_rss_mib(),
    )


__all__ = [
    "ATTRITION_SCHEMA",
    "BATCH_MANIFEST_SCHEMA",
    "BUILDER_VERSION",
    "DEFAULT_CONTEXT_MANIFEST",
    "DEFAULT_CONTEXT_MANIFEST_SHA256",
    "DEFAULT_FINALIZED_EPISODE_MANIFEST",
    "DEFAULT_FINALIZED_EPISODE_MANIFEST_SHA256",
    "DEFAULT_OUTPUT_ROOT",
    "ENDPOINT_SECONDS",
    "FEATURE_BLOCKS",
    "LANDMARK_SECONDS",
    "RegulationGamePanel",
    "RegulationPanelError",
    "RegulationPanelSourceSpec",
    "PublishedRegulationPanel",
    "build_regulation_game_panel",
    "default_source_spec",
    "publish_exact153_regulation_panel",
    "verify_regulation_panel_sources",
]
