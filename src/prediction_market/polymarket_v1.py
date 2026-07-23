"""Bounded, revision-locked access POC for TimeSeventeen/Polymarket-v1."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import quote

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from prediction_market.static_store import StaticObjectRecord, preserve_static_object


POLYMARKET_V1_DATASET = "TimeSeventeen/Polymarket-v1"
POLYMARKET_V1_DATASET_ID = "DS-POLYMARKET-V1"
POLYMARKET_V1_REVISION = "66a1d6ddfc3cdab9e2087c1e2e855bab272d3404"
POLYMARKET_V1_LICENSE_REF = "R-039"
POLYMARKET_V1_LICENSE_ID = "CC-BY-4.0"
HF_REPO_API_ROOT = (
    "https://huggingface.co/api/datasets/TimeSeventeen/Polymarket-v1"
)
HF_DATASET_SERVER_ROOT = "https://datasets-server.huggingface.co"
HF_RESOLVE_ROOT = (
    "https://huggingface.co/datasets/TimeSeventeen/Polymarket-v1/resolve/"
    + POLYMARKET_V1_REVISION
)
GAMMA_MARKET_SLUG_ROOT = "https://gamma-api.polymarket.com/markets/slug"

BOUNDED_SHARD_PATH = "daily_aligned/2023-01-01.parquet"
BOUNDED_SHARD_SIZE = 13_454
BOUNDED_SHARD_SHA256 = (
    "sha256:7a5b235e35fbe8c0b431ac88eab3464776f07180ca2f32c59d4ab0da7a493cbe"
)
BOUNDED_SHARD_EXPECTED_ROWS = 31
BOUNDED_EXTRACT_EXPECTED_ROWS = 17

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONDITION_ID_RE = re.compile(r"^0x[0-9a-f]{64}$")
_REQUIRED_DAILY_COLUMNS = (
    "asset_id",
    "block_timestamp",
    "price",
    "maker",
    "taker",
    "taker_direction",
    "usdc_amount",
    "fee_usdc",
    "condition_id",
    "outcome_seq",
    "neg_risk",
    "category",
    "category_refined",
    "outcome_label",
    "winning_outcome_label",
    "resolution_status",
    "taker_base_fee",
    "maker_base_fee",
    "opens_at",
    "close_at",
    "resolved_at",
    "market_slug",
    "p_event",
    "D",
)


class PolymarketV1SourceError(ValueError):
    """A source object cannot support the bounded Polymarket-v1 POC."""


@dataclass(frozen=True, slots=True)
class FrozenSportsCondition:
    condition_id: str
    slug: str
    gamma_market_id: str
    sport: str = "NFL"


FROZEN_NFL_CONDITIONS = (
    FrozenSportsCondition(
        condition_id=(
            "0x86a95921efba50773e078f7b935e0ce94b6bfb35f42fcab86ff63ff226d9237f"
        ),
        slug="nfl-sunday-dolphins-vs-patriots",
        gamma_market_id="248292",
    ),
    FrozenSportsCondition(
        condition_id=(
            "0x9a7209897461d86a93aec03b1f731c338b43ef3a8306f2b8a2b1f295e04f193f"
        ),
        slug="nfl-sunday-vikings-vs-packers",
        gamma_market_id="248293",
    ),
    FrozenSportsCondition(
        condition_id=(
            "0x5e5172329111b4ee0c810f2c96f30a5967e9dd283ed8a4f27fff5056246b2141"
        ),
        slug="nfl-sunday-steelers-vs-ravens",
        gamma_market_id="248294",
    ),
    FrozenSportsCondition(
        condition_id=(
            "0x80dcb010ea53350c0a67f018c861d11239e5638de7d7c779510a7177cf1a1974"
        ),
        slug="nfl-monday-bills-vs-bengals",
        gamma_market_id="248295",
    ),
)
FROZEN_NFL_CONDITION_BY_ID = MappingProxyType(
    {condition.condition_id: condition for condition in FROZEN_NFL_CONDITIONS}
)


@dataclass(frozen=True, slots=True)
class RepoMetadataAudit:
    revision: str
    license_id: str
    private: bool
    gated: bool
    schema_fingerprint: str
    object_sha256: str


@dataclass(frozen=True, slots=True)
class RepoTreeAudit:
    shard_path: str
    shard_size: int
    shard_sha256: str
    entry_count: int
    schema_fingerprint: str
    object_sha256: str


@dataclass(frozen=True, slots=True)
class ParquetIndexAudit:
    file_count: int
    total_bytes: int
    configs: tuple[str, ...]
    schema_fingerprint: str
    object_sha256: str


@dataclass(frozen=True, slots=True)
class FirstRowsAudit:
    row_count: int
    feature_names: tuple[str, ...]
    condition_count: int
    sample_truncated: bool
    schema_fingerprint: str
    object_sha256: str


@dataclass(frozen=True, slots=True)
class DailyAlignedShardAudit:
    row_count: int
    condition_count: int
    column_names: tuple[str, ...]
    first_block_timestamp: int
    last_block_timestamp: int
    schema_fingerprint: str
    object_sha256: str


@dataclass(frozen=True, slots=True)
class GammaMarketAudit:
    market_id: str
    condition_id: str
    slug: str
    sport: str
    schema_fingerprint: str
    object_sha256: str


@dataclass(frozen=True, slots=True)
class ExactConditionExtract:
    payload: bytes
    row_count: int
    condition_count: int
    matched_condition_ids: tuple[str, ...]
    query: Mapping[str, Any]
    query_sha256: str
    lineage_refs: tuple[str, ...]
    schema_fingerprint: str
    object_sha256: str
    pit_safe_for_model_features: bool
    contains_l2: bool


@dataclass(frozen=True, slots=True)
class PreservedDailyAlignedShard:
    record: StaticObjectRecord
    audit: DailyAlignedShardAudit


@dataclass(frozen=True, slots=True)
class PreservedGammaMarket:
    record: StaticObjectRecord
    audit: GammaMarketAudit


@dataclass(frozen=True, slots=True)
class PreservedSportsExtract:
    record: StaticObjectRecord
    audit: ExactConditionExtract


@dataclass(frozen=True, slots=True)
class PolymarketV1Capture:
    repo_metadata: StaticObjectRecord
    repo_tree: StaticObjectRecord
    parquet_index: StaticObjectRecord
    first_rows: StaticObjectRecord
    source_shard: PreservedDailyAlignedShard
    gamma_markets: tuple[PreservedGammaMarket, ...]
    sports_extract: PreservedSportsExtract
    repo_metadata_audit: RepoMetadataAudit
    repo_tree_audit: RepoTreeAudit
    parquet_index_audit: ParquetIndexAudit
    first_rows_audit: FirstRowsAudit


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PolymarketV1SourceError("value is not canonical JSON") from exc


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolymarketV1SourceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(payload: bytes, *, context: str) -> Any:
    if type(payload) is not bytes or not payload:
        raise PolymarketV1SourceError(f"{context} must contain exact nonempty bytes")
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                PolymarketV1SourceError(f"non-finite JSON value: {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, PolymarketV1SourceError) as exc:
        raise PolymarketV1SourceError(
            f"{context} is not strict UTF-8 JSON"
        ) from exc


def _required_object(value: Any, field: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise PolymarketV1SourceError(f"{field} must be a JSON object")
    return value


def _required_text(value: Any, field: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise PolymarketV1SourceError(f"{field} must be a nonempty string")
    return value


def _json_schema_fingerprint(value: Any) -> str:
    observed: dict[str, set[str]] = defaultdict(set)

    def kind(child: Any) -> str:
        if child is None:
            return "null"
        if type(child) is bool:
            return "boolean"
        if type(child) is int:
            return "integer"
        if type(child) is float:
            return "number"
        if type(child) is str:
            return "string"
        if type(child) is list:
            return "array"
        if type(child) is dict:
            return "object"
        raise PolymarketV1SourceError("JSON contains an unsupported value type")

    def visit(child: Any, path: str) -> None:
        observed[path].add(kind(child))
        if type(child) is dict:
            for key in sorted(child):
                visit(child[key], f"{path}.{key}")
        elif type(child) is list:
            for item in child:
                visit(item, f"{path}[]")

    visit(value, "$")
    material = {path: sorted(types) for path, types in sorted(observed.items())}
    return "sha256:" + hashlib.sha256(_canonical_bytes(material)).hexdigest()


def _object_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _arrow_schema_fingerprint(schema: pa.Schema) -> str:
    return "sha256:" + hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _require_revision_header(value: str | None) -> None:
    if value != POLYMARKET_V1_REVISION:
        raise PolymarketV1SourceError(
            "dataset-server X-Revision does not match the frozen revision"
        )


def inspect_repo_metadata(payload: bytes) -> RepoMetadataAudit:
    """Validate the repository identity, revision, and declared dataset license."""

    metadata = _required_object(
        _strict_json(payload, context="Hugging Face repository metadata"),
        "repository metadata",
    )
    if metadata.get("id") != POLYMARKET_V1_DATASET:
        raise PolymarketV1SourceError("repository id does not match Polymarket-v1")
    revision = _required_text(metadata.get("sha"), "repository sha")
    if revision != POLYMARKET_V1_REVISION:
        raise PolymarketV1SourceError("repository revision is not frozen")
    if metadata.get("private") is not False or metadata.get("gated") is not False:
        raise PolymarketV1SourceError("repository must be public and ungated")
    tags = metadata.get("tags")
    if type(tags) is not list or any(type(tag) is not str for tag in tags):
        raise PolymarketV1SourceError("repository tags must be strings")
    if "license:cc-by-4.0" not in tags:
        raise PolymarketV1SourceError("repository does not declare CC-BY-4.0")
    return RepoMetadataAudit(
        revision=revision,
        license_id=POLYMARKET_V1_LICENSE_ID,
        private=False,
        gated=False,
        schema_fingerprint=_json_schema_fingerprint(metadata),
        object_sha256=_object_sha256(payload),
    )


def inspect_repo_tree(payload: bytes) -> RepoTreeAudit:
    """Find the one canonical bounded shard in a revision-locked tree page."""

    tree = _strict_json(payload, context="Hugging Face repository tree")
    if type(tree) is not list or not tree:
        raise PolymarketV1SourceError("repository tree must be a nonempty array")
    candidates = [
        entry
        for entry in tree
        if type(entry) is dict and entry.get("path") == BOUNDED_SHARD_PATH
    ]
    if len(candidates) != 1:
        raise PolymarketV1SourceError(
            "repository tree must contain the bounded shard exactly once"
        )
    entry = candidates[0]
    if entry.get("type") != "file":
        raise PolymarketV1SourceError("bounded shard tree entry is not a file")
    size = entry.get("size")
    lfs = _required_object(entry.get("lfs"), "bounded shard lfs metadata")
    lfs_size = lfs.get("size")
    lfs_oid = lfs.get("oid")
    if (
        type(size) is not int
        or size <= 0
        or type(lfs_size) is not int
        or lfs_size != size
        or type(lfs_oid) is not str
        or re.fullmatch(r"[0-9a-f]{64}", lfs_oid) is None
    ):
        raise PolymarketV1SourceError("bounded shard LFS size/hash is invalid")
    return RepoTreeAudit(
        shard_path=BOUNDED_SHARD_PATH,
        shard_size=size,
        shard_sha256=f"sha256:{lfs_oid}",
        entry_count=len(tree),
        schema_fingerprint=_json_schema_fingerprint(tree),
        object_sha256=_object_sha256(payload),
    )


def inspect_parquet_index(
    payload: bytes, *, response_revision: str | None
) -> ParquetIndexAudit:
    """Validate the exact-revision dataset-server Parquet inventory."""

    _require_revision_header(response_revision)
    document = _required_object(
        _strict_json(payload, context="Hugging Face Parquet index"),
        "Parquet index",
    )
    files = document.get("parquet_files")
    if type(files) is not list or not files:
        raise PolymarketV1SourceError("Parquet index must list files")
    configs: set[str] = set()
    total_bytes = 0
    for position, raw_file in enumerate(files):
        item = _required_object(raw_file, f"parquet_files[{position}]")
        if item.get("dataset") != POLYMARKET_V1_DATASET:
            raise PolymarketV1SourceError("Parquet file has the wrong dataset")
        config = _required_text(item.get("config"), "Parquet config")
        if item.get("split") != "train":
            raise PolymarketV1SourceError("Parquet split must be train")
        url = _required_text(item.get("url"), "Parquet URL")
        if not url.startswith(
            "https://huggingface.co/datasets/TimeSeventeen/Polymarket-v1/"
        ):
            raise PolymarketV1SourceError("Parquet URL is outside the dataset")
        size = item.get("size")
        if type(size) is not int or size <= 0:
            raise PolymarketV1SourceError("Parquet size must be positive")
        configs.add(config)
        total_bytes += size
    required_configs = {"ctf", "daily_aligned", "orderfilled"}
    if configs != required_configs:
        raise PolymarketV1SourceError(
            "Parquet index must expose ctf, daily_aligned, and orderfilled"
        )
    for state in ("pending", "failed"):
        if document.get(state, []) != []:
            raise PolymarketV1SourceError(
                f"Parquet index has unresolved {state} conversions"
            )
    return ParquetIndexAudit(
        file_count=len(files),
        total_bytes=total_bytes,
        configs=tuple(sorted(configs)),
        schema_fingerprint=_json_schema_fingerprint(document),
        object_sha256=_object_sha256(payload),
    )


def inspect_first_rows_response(
    payload: bytes, *, response_revision: str | None
) -> FirstRowsAudit:
    """Validate the revision-bound daily_aligned schema/first-rows response."""

    _require_revision_header(response_revision)
    document = _required_object(
        _strict_json(payload, context="Hugging Face first-rows response"),
        "first-rows response",
    )
    if (
        document.get("dataset") != POLYMARKET_V1_DATASET
        or document.get("config") != "daily_aligned"
        or document.get("split") != "train"
    ):
        raise PolymarketV1SourceError(
            "first-rows response is not daily_aligned/train"
        )
    features = document.get("features")
    if type(features) is not list or not features:
        raise PolymarketV1SourceError("first-rows features must be nonempty")
    names: list[str] = []
    for index, raw_feature in enumerate(features):
        feature = _required_object(raw_feature, f"features[{index}]")
        if feature.get("feature_idx") != index:
            raise PolymarketV1SourceError("feature_idx must be source ordered")
        names.append(_required_text(feature.get("name"), "feature name"))
    if len(set(names)) != len(names):
        raise PolymarketV1SourceError("first-rows feature names must be unique")
    missing = sorted(set(_REQUIRED_DAILY_COLUMNS) - set(names))
    if missing:
        raise PolymarketV1SourceError(
            "first-rows schema is missing required field(s): "
            + ",".join(missing)
        )
    rows = document.get("rows")
    if type(rows) is not list or not rows:
        raise PolymarketV1SourceError("first-rows response must contain rows")
    condition_ids: set[str] = set()
    for index, raw_envelope in enumerate(rows):
        envelope = _required_object(raw_envelope, f"rows[{index}]")
        if envelope.get("row_idx") != index:
            raise PolymarketV1SourceError("row_idx must be source ordered")
        if envelope.get("truncated_cells") != []:
            raise PolymarketV1SourceError("first-rows cells must not be truncated")
        row = _required_object(envelope.get("row"), f"rows[{index}].row")
        if set(row) != set(names):
            raise PolymarketV1SourceError(
                "first-rows row fields do not match features"
            )
        condition_id = _required_text(row.get("condition_id"), "condition_id")
        if _CONDITION_ID_RE.fullmatch(condition_id) is None:
            raise PolymarketV1SourceError("condition_id is not canonical")
        condition_ids.add(condition_id)
    sample_truncated = document.get("truncated")
    if type(sample_truncated) is not bool:
        raise PolymarketV1SourceError(
            "first-rows truncated flag must be boolean"
        )
    return FirstRowsAudit(
        row_count=len(rows),
        feature_names=tuple(names),
        condition_count=len(condition_ids),
        sample_truncated=sample_truncated,
        schema_fingerprint=_json_schema_fingerprint(document),
        object_sha256=_object_sha256(payload),
    )


def _read_daily_table(payload: bytes) -> pa.Table:
    if type(payload) is not bytes or not payload:
        raise PolymarketV1SourceError(
            "daily_aligned shard must contain exact nonempty bytes"
        )
    try:
        table = pq.read_table(pa.BufferReader(payload))
    except (pa.ArrowException, OSError, ValueError) as exc:
        raise PolymarketV1SourceError(
            "daily_aligned shard is not readable Parquet"
        ) from exc
    if table.num_rows <= 0:
        raise PolymarketV1SourceError("daily_aligned shard must contain rows")
    missing = sorted(set(_REQUIRED_DAILY_COLUMNS) - set(table.column_names))
    if missing:
        raise PolymarketV1SourceError(
            "daily_aligned shard is missing field(s): " + ",".join(missing)
        )
    if len(table.column_names) != len(set(table.column_names)):
        raise PolymarketV1SourceError("daily_aligned column names are duplicated")
    return table


def _validated_daily_rows(table: pa.Table) -> list[dict[str, Any]]:
    rows = table.to_pylist()
    for index, row in enumerate(rows):
        condition_id = row.get("condition_id")
        if (
            type(condition_id) is not str
            or _CONDITION_ID_RE.fullmatch(condition_id) is None
        ):
            raise PolymarketV1SourceError(
                f"row {index} condition_id is not canonical"
            )
        block_timestamp = row.get("block_timestamp")
        if type(block_timestamp) is not int or block_timestamp <= 0:
            raise PolymarketV1SourceError(
                f"row {index} block_timestamp must be a positive Unix second"
            )
        price = row.get("price")
        if (
            type(price) not in {int, float}
            or not math.isfinite(float(price))
            or not 0 <= float(price) <= 1
        ):
            raise PolymarketV1SourceError(f"row {index} price is invalid")
        amount = row.get("usdc_amount")
        if (
            type(amount) not in {int, float}
            or not math.isfinite(float(amount))
            or float(amount) < 0
        ):
            raise PolymarketV1SourceError(
                f"row {index} usdc_amount is invalid"
            )
    return rows


def inspect_daily_aligned_shard(
    payload: bytes, *, expected_size: int, expected_sha256: str
) -> DailyAlignedShardAudit:
    """Validate a bounded source Parquet against its exact tree LFS lock."""

    if type(expected_size) is not int or expected_size <= 0:
        raise PolymarketV1SourceError("expected_size must be positive")
    if (
        type(expected_sha256) is not str
        or _SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise PolymarketV1SourceError("expected_sha256 must be canonical")
    if len(payload) != expected_size:
        raise PolymarketV1SourceError("daily_aligned shard size mismatch")
    object_hash = _object_sha256(payload)
    if object_hash != expected_sha256:
        raise PolymarketV1SourceError("daily_aligned shard SHA-256 mismatch")
    table = _read_daily_table(payload)
    rows = _validated_daily_rows(table)
    timestamps = [row["block_timestamp"] for row in rows]
    return DailyAlignedShardAudit(
        row_count=len(rows),
        condition_count=len({row["condition_id"] for row in rows}),
        column_names=tuple(table.column_names),
        first_block_timestamp=min(timestamps),
        last_block_timestamp=max(timestamps),
        schema_fingerprint=_arrow_schema_fingerprint(table.schema),
        object_sha256=object_hash,
    )


def inspect_gamma_market(
    payload: bytes, *, expected: FrozenSportsCondition
) -> GammaMarketAudit:
    """Prove that official Gamma metadata maps the frozen ID to an NFL event."""

    if not isinstance(expected, FrozenSportsCondition):
        raise TypeError("expected must be a FrozenSportsCondition")
    market = _required_object(
        _strict_json(payload, context="Gamma market response"),
        "Gamma market response",
    )
    if market.get("id") != expected.gamma_market_id:
        raise PolymarketV1SourceError("Gamma market id does not match frozen mapping")
    if market.get("conditionId") != expected.condition_id:
        raise PolymarketV1SourceError(
            "Gamma conditionId does not match frozen mapping"
        )
    if market.get("slug") != expected.slug:
        raise PolymarketV1SourceError("Gamma slug does not match frozen mapping")
    events = market.get("events")
    if type(events) is not list or not events:
        raise PolymarketV1SourceError("Gamma market has no event metadata")
    verified_nfl = False
    for index, raw_event in enumerate(events):
        event = _required_object(raw_event, f"Gamma events[{index}]")
        if event.get("seriesSlug") == "nfl":
            verified_nfl = True
    if not verified_nfl:
        raise PolymarketV1SourceError(
            "Gamma metadata does not verify the frozen market as NFL"
        )
    return GammaMarketAudit(
        market_id=expected.gamma_market_id,
        condition_id=expected.condition_id,
        slug=expected.slug,
        sport=expected.sport,
        schema_fingerprint=_json_schema_fingerprint(market),
        object_sha256=_object_sha256(payload),
    )


def _validated_lineage_ref(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise PolymarketV1SourceError(f"{field} must be a canonical SHA-256")
    return value


def _json_safe_arrow_value(value: Any, *, field: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PolymarketV1SourceError(f"{field} contains a non-finite float")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise PolymarketV1SourceError(
                f"{field} contains a timezone-naive timestamp"
            )
        return (
            value.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if type(value) is list:
        return [
            _json_safe_arrow_value(child, field=f"{field}[]")
            for child in value
        ]
    if type(value) is dict:
        return {
            str(key): _json_safe_arrow_value(
                child, field=f"{field}.{key}"
            )
            for key, child in value.items()
        }
    raise PolymarketV1SourceError(
        f"{field} has unsupported Arrow value {type(value).__name__}"
    )


def build_exact_condition_extract(
    source_payload: bytes,
    *,
    source_object_ref: str,
    gamma_object_refs: Sequence[str],
) -> ExactConditionExtract:
    """Filter only by equality against the frozen condition_id registry."""

    source_ref = _validated_lineage_ref(source_object_ref, "source_object_ref")
    refs = tuple(
        _validated_lineage_ref(value, f"gamma_object_refs[{index}]")
        for index, value in enumerate(gamma_object_refs)
    )
    if len(refs) != len(FROZEN_NFL_CONDITIONS) or len(set(refs)) != len(refs):
        raise PolymarketV1SourceError(
            "one unique Gamma object ref is required per frozen condition"
        )
    if source_ref in refs:
        raise PolymarketV1SourceError("source and Gamma lineage refs must differ")
    table = _read_daily_table(source_payload)
    rows = _validated_daily_rows(table)
    condition_ids = frozenset(FROZEN_NFL_CONDITION_BY_ID)
    selected = [
        {
            "source_row_index": index,
            "row": {
                key: _json_safe_arrow_value(
                    value, field=f"row[{index}].{key}"
                )
                for key, value in row.items()
            },
        }
        for index, row in enumerate(rows)
        if row["condition_id"] in condition_ids
    ]
    matched = tuple(
        sorted({item["row"]["condition_id"] for item in selected})
    )
    if matched != tuple(sorted(condition_ids)):
        missing = sorted(condition_ids - set(matched))
        raise PolymarketV1SourceError(
            "bounded shard is missing frozen condition_id(s): " + ",".join(missing)
        )
    query: dict[str, Any] = {
        "query_version": "v0",
        "operation": "exact_condition_id_inner_join",
        "source_dataset_id": POLYMARKET_V1_DATASET_ID,
        "source_revision": POLYMARKET_V1_REVISION,
        "source_partition": BOUNDED_SHARD_PATH,
        "join_key": "condition_id",
        "predicate_fields": ["condition_id"],
        "condition_ids": sorted(condition_ids),
        "gamma_verification": [
            {
                "condition_id": condition.condition_id,
                "gamma_market_id": condition.gamma_market_id,
                "slug": condition.slug,
                "sport": condition.sport,
                "object_sha256": ref,
            }
            for condition, ref in zip(FROZEN_NFL_CONDITIONS, refs, strict=True)
        ],
        "source_order_preserved": True,
        "timestamp_encoding": "UTC RFC3339 with six fractional digits",
        "point_in_time_feature_eligibility": False,
    }
    query_sha256 = "sha256:" + hashlib.sha256(_canonical_bytes(query)).hexdigest()
    lines = [_canonical_bytes(item) for item in selected]
    payload = b"\n".join(lines) + b"\n"
    output_schema = {
        "envelope": {
            "source_row_index": "integer",
            "row": {
                field.name: str(field.type)
                for field in table.schema
            },
        }
    }
    return ExactConditionExtract(
        payload=payload,
        row_count=len(selected),
        condition_count=len(matched),
        matched_condition_ids=matched,
        query=MappingProxyType(query),
        query_sha256=query_sha256,
        lineage_refs=(source_ref, *refs),
        schema_fingerprint=(
            "sha256:" + hashlib.sha256(_canonical_bytes(output_schema)).hexdigest()
        ),
        object_sha256=_object_sha256(payload),
        pit_safe_for_model_features=False,
        contains_l2=False,
    )


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PolymarketV1SourceError("fetched_at must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _download(
    url: str,
    *,
    params: Mapping[str, object] | None,
    client: httpx.Client,
    max_bytes: int,
) -> tuple[bytes, httpx.Headers]:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise PolymarketV1SourceError("max_bytes must be a positive integer")
    try:
        with client.stream(
            "GET",
            url,
            params=params,
            headers={"Accept": "*/*", "Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            encoding = response.headers.get("Content-Encoding")
            if encoding not in {None, "identity"}:
                raise PolymarketV1SourceError(
                    "source ignored identity content encoding"
                )
            declared_text = response.headers.get("Content-Length")
            declared: int | None = None
            if declared_text is not None:
                try:
                    declared = int(declared_text)
                except ValueError as exc:
                    raise PolymarketV1SourceError(
                        "source returned invalid Content-Length"
                    ) from exc
                if declared < 0 or declared > max_bytes:
                    raise PolymarketV1SourceError(
                        "source response exceeds max_bytes"
                    )
            chunks: list[bytes] = []
            received = 0
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > max_bytes:
                    raise PolymarketV1SourceError(
                        "source response exceeds max_bytes"
                    )
                chunks.append(chunk)
            if declared is not None and declared != received:
                raise PolymarketV1SourceError(
                    "source Content-Length does not match received bytes"
                )
            payload = b"".join(chunks)
            headers = httpx.Headers(response.headers)
    except httpx.HTTPError as exc:
        raise PolymarketV1SourceError(f"source request failed: {exc}") from exc
    if not payload:
        raise PolymarketV1SourceError("source request returned empty bytes")
    return payload, headers


def _active_client(client: httpx.Client | None) -> tuple[httpx.Client, bool]:
    if client is not None:
        return client, False
    return (
        httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(120.0, connect=30.0),
        ),
        True,
    )


def _content_type(headers: httpx.Headers, default: str) -> str:
    return headers.get("Content-Type", default).split(";", 1)[0]


def _preserve_hf_json(
    payload: bytes,
    headers: httpx.Headers,
    *,
    store_root: str | Path,
    program_root: str | Path,
    partition: str,
    source_url: str,
    params: Mapping[str, object],
    source_cursor: str,
    fetched_at: str,
    coverage: str,
    schema_fingerprint: str,
) -> StaticObjectRecord:
    return preserve_static_object(
        store_root,
        payload,
        program_root=program_root,
        source="huggingface",
        dataset=POLYMARKET_V1_DATASET_ID,
        version=POLYMARKET_V1_REVISION,
        partition=partition,
        extension="json",
        source_url=source_url,
        source_request={
            "method": "GET",
            "headers": {"Accept": "*/*", "Accept-Encoding": "identity"},
            "params": dict(params),
        },
        source_cursor=source_cursor,
        fetched_at=fetched_at,
        coverage=coverage,
        etag=headers.get("ETag"),
        last_modified=headers.get("Last-Modified"),
        media_type=_content_type(headers, "application/json"),
        schema_fingerprint=schema_fingerprint,
        license_ref=POLYMARKET_V1_LICENSE_REF,
        license_status="approved",
        upstream_partition=partition,
        object_kind="byte_exact_original",
        lineage={"source_object_refs": [], "query_sha256": None},
    )


def capture_polymarket_v1_poc(
    *,
    store_root: str | Path,
    program_root: str | Path,
    fetched_at: datetime,
    client: httpx.Client | None = None,
) -> PolymarketV1Capture:
    """Capture official bounded evidence and publish one exact-ID sports extract."""

    fetched_at_text = _utc_text(fetched_at)
    active, owned = _active_client(client)
    metadata_url = f"{HF_REPO_API_ROOT}/revision/{POLYMARKET_V1_REVISION}"
    tree_url = f"{HF_REPO_API_ROOT}/tree/{POLYMARKET_V1_REVISION}"
    parquet_url = f"{HF_DATASET_SERVER_ROOT}/parquet"
    first_rows_url = f"{HF_DATASET_SERVER_ROOT}/first-rows"
    shard_url = f"{HF_RESOLVE_ROOT}/{BOUNDED_SHARD_PATH}"
    tree_params: dict[str, object] = {
        "recursive": "true",
        "expand": "false",
        "limit": 1000,
    }
    revision_params: dict[str, object] = {
        "dataset": POLYMARKET_V1_DATASET,
        "revision": POLYMARKET_V1_REVISION,
    }
    first_rows_params: dict[str, object] = {
        **revision_params,
        "config": "daily_aligned",
        "split": "train",
    }
    try:
        metadata_payload, metadata_headers = _download(
            metadata_url, params=None, client=active, max_bytes=2_000_000
        )
        metadata_audit = inspect_repo_metadata(metadata_payload)
        metadata_record = _preserve_hf_json(
            metadata_payload,
            metadata_headers,
            store_root=store_root,
            program_root=program_root,
            partition="repo-metadata",
            source_url=metadata_url,
            params={},
            source_cursor=f"revision:{POLYMARKET_V1_REVISION}",
            fetched_at=fetched_at_text,
            coverage="revision-locked repository metadata and declared license",
            schema_fingerprint=metadata_audit.schema_fingerprint,
        )

        tree_payload, tree_headers = _download(
            tree_url,
            params=tree_params,
            client=active,
            max_bytes=2_000_000,
        )
        tree_audit = inspect_repo_tree(tree_payload)
        if (
            tree_audit.shard_size != BOUNDED_SHARD_SIZE
            or tree_audit.shard_sha256 != BOUNDED_SHARD_SHA256
        ):
            raise PolymarketV1SourceError(
                "bounded shard tree lock differs from the approved POC"
            )
        tree_record = _preserve_hf_json(
            tree_payload,
            tree_headers,
            store_root=store_root,
            program_root=program_root,
            partition="repo-tree-0001",
            source_url=tree_url,
            params=tree_params,
            source_cursor="tree_cursor:ROOT;bounded_page:true",
            fetched_at=fetched_at_text,
            coverage=(
                "bounded first tree page; selected "
                + BOUNDED_SHARD_PATH
                + ";not a complete repository inventory"
            ),
            schema_fingerprint=tree_audit.schema_fingerprint,
        )

        parquet_payload, parquet_headers = _download(
            parquet_url,
            params=revision_params,
            client=active,
            max_bytes=2_000_000,
        )
        parquet_audit = inspect_parquet_index(
            parquet_payload,
            response_revision=parquet_headers.get("X-Revision"),
        )
        parquet_record = _preserve_hf_json(
            parquet_payload,
            parquet_headers,
            store_root=store_root,
            program_root=program_root,
            partition="parquet-index",
            source_url=parquet_url,
            params=revision_params,
            source_cursor=f"revision:{POLYMARKET_V1_REVISION}",
            fetched_at=fetched_at_text,
            coverage="all dataset-server Parquet conversion entries",
            schema_fingerprint=parquet_audit.schema_fingerprint,
        )

        first_rows_payload, first_rows_headers = _download(
            first_rows_url,
            params=first_rows_params,
            client=active,
            max_bytes=2_000_000,
        )
        first_rows_audit = inspect_first_rows_response(
            first_rows_payload,
            response_revision=first_rows_headers.get("X-Revision"),
        )
        first_rows_record = _preserve_hf_json(
            first_rows_payload,
            first_rows_headers,
            store_root=store_root,
            program_root=program_root,
            partition="first-rows-daily-aligned",
            source_url=first_rows_url,
            params=first_rows_params,
            source_cursor=f"revision:{POLYMARKET_V1_REVISION};row_offset:0",
            fetched_at=fetched_at_text,
            coverage="dataset-server daily_aligned/train schema and first rows",
            schema_fingerprint=first_rows_audit.schema_fingerprint,
        )

        shard_payload, shard_headers = _download(
            shard_url,
            params=None,
            client=active,
            max_bytes=1_000_000,
        )
        shard_audit = inspect_daily_aligned_shard(
            shard_payload,
            expected_size=tree_audit.shard_size,
            expected_sha256=tree_audit.shard_sha256,
        )
        if shard_audit.row_count != BOUNDED_SHARD_EXPECTED_ROWS:
            raise PolymarketV1SourceError(
                "bounded shard row count differs from the approved POC"
            )
        shard_partition = "daily-aligned-2023-01-01"
        shard_record = preserve_static_object(
            store_root,
            shard_payload,
            program_root=program_root,
            source="huggingface",
            dataset=POLYMARKET_V1_DATASET_ID,
            version=POLYMARKET_V1_REVISION,
            partition=shard_partition,
            extension="parquet",
            source_url=shard_url,
            source_request={
                "method": "GET",
                "headers": {"Accept": "*/*", "Accept-Encoding": "identity"},
                "tree_manifest_sha256": tree_record.manifest.manifest_sha256,
            },
            source_cursor=(
                f"revision:{POLYMARKET_V1_REVISION};path:{BOUNDED_SHARD_PATH}"
            ),
            fetched_at=fetched_at_text,
            coverage="UTC source partition 2023-01-01;bounded schema smoke",
            etag=shard_headers.get("ETag"),
            last_modified=shard_headers.get("Last-Modified"),
            media_type=_content_type(
                shard_headers, "application/vnd.apache.parquet"
            ),
            schema_fingerprint=shard_audit.schema_fingerprint,
            license_ref=POLYMARKET_V1_LICENSE_REF,
            license_status="approved",
            upstream_partition=shard_partition,
            object_kind="byte_exact_original",
            lineage={"source_object_refs": [], "query_sha256": None},
        )

        gamma_markets: list[PreservedGammaMarket] = []
        for condition in FROZEN_NFL_CONDITIONS:
            gamma_url = f"{GAMMA_MARKET_SLUG_ROOT}/{quote(condition.slug)}"
            gamma_payload, gamma_headers = _download(
                gamma_url, params=None, client=active, max_bytes=1_000_000
            )
            gamma_audit = inspect_gamma_market(
                gamma_payload, expected=condition
            )
            gamma_partition = f"gamma-market-{condition.gamma_market_id}"
            gamma_record = preserve_static_object(
                store_root,
                gamma_payload,
                program_root=program_root,
                source="polymarket",
                dataset="DS-POLYMARKET-PUBLIC",
                version="public-api-20260722",
                partition=gamma_partition,
                extension="json",
                source_url=gamma_url,
                source_request={
                    "method": "GET",
                    "headers": {"Accept": "*/*", "Accept-Encoding": "identity"},
                    "params": {},
                },
                source_cursor=(
                    f"market_id:{condition.gamma_market_id};"
                    f"condition_id:{condition.condition_id}"
                ),
                fetched_at=fetched_at_text,
                coverage=(
                    "point-in-time exact condition_id and NFL series validation"
                ),
                etag=gamma_headers.get("ETag"),
                last_modified=gamma_headers.get("Last-Modified"),
                media_type=_content_type(gamma_headers, "application/json"),
                schema_fingerprint=gamma_audit.schema_fingerprint,
                license_ref="O-001",
                license_status="pending",
                upstream_partition=gamma_partition,
                object_kind="byte_exact_original",
                lineage={"source_object_refs": [], "query_sha256": None},
            )
            gamma_markets.append(
                PreservedGammaMarket(record=gamma_record, audit=gamma_audit)
            )

        extract_audit = build_exact_condition_extract(
            shard_payload,
            source_object_ref=shard_record.manifest.object_sha256,
            gamma_object_refs=tuple(
                item.record.manifest.object_sha256 for item in gamma_markets
            ),
        )
        if extract_audit.row_count != BOUNDED_EXTRACT_EXPECTED_ROWS:
            raise PolymarketV1SourceError(
                "bounded sports extract row count differs from the approved POC"
            )
        extract_partition = "sports-extract-2023-01-01"
        extract_record = preserve_static_object(
            store_root,
            extract_audit.payload,
            program_root=program_root,
            source="huggingface",
            dataset=POLYMARKET_V1_DATASET_ID,
            version=POLYMARKET_V1_REVISION,
            partition=extract_partition,
            extension="jsonl",
            source_url=shard_url,
            source_request={
                "method": "DERIVE",
                "query": dict(extract_audit.query),
                "source_manifest_sha256": (
                    shard_record.manifest.manifest_sha256
                ),
                "gamma_manifest_sha256": [
                    item.record.manifest.manifest_sha256
                    for item in gamma_markets
                ],
            },
            source_cursor=(
                f"revision:{POLYMARKET_V1_REVISION};"
                f"query_sha256:{extract_audit.query_sha256}"
            ),
            fetched_at=fetched_at_text,
            coverage=(
                "bounded 2023-01-01 NFL fills exact-joined by frozen condition_id"
            ),
            etag=None,
            last_modified=None,
            media_type="application/x-ndjson",
            schema_fingerprint=extract_audit.schema_fingerprint,
            license_ref=POLYMARKET_V1_LICENSE_REF,
            license_status="approved",
            upstream_partition=extract_partition,
            object_kind="source_derived_extract",
            lineage={
                "source_object_refs": list(extract_audit.lineage_refs),
                "query_sha256": extract_audit.query_sha256,
            },
        )
    finally:
        if owned:
            active.close()

    return PolymarketV1Capture(
        repo_metadata=metadata_record,
        repo_tree=tree_record,
        parquet_index=parquet_record,
        first_rows=first_rows_record,
        source_shard=PreservedDailyAlignedShard(
            record=shard_record, audit=shard_audit
        ),
        gamma_markets=tuple(gamma_markets),
        sports_extract=PreservedSportsExtract(
            record=extract_record, audit=extract_audit
        ),
        repo_metadata_audit=metadata_audit,
        repo_tree_audit=tree_audit,
        parquet_index_audit=parquet_audit,
        first_rows_audit=first_rows_audit,
    )


__all__ = [
    "BOUNDED_SHARD_PATH",
    "FROZEN_NFL_CONDITIONS",
    "POLYMARKET_V1_REVISION",
    "DailyAlignedShardAudit",
    "ExactConditionExtract",
    "FirstRowsAudit",
    "FrozenSportsCondition",
    "GammaMarketAudit",
    "ParquetIndexAudit",
    "PolymarketV1Capture",
    "PolymarketV1SourceError",
    "RepoMetadataAudit",
    "RepoTreeAudit",
    "build_exact_condition_extract",
    "capture_polymarket_v1_poc",
    "inspect_daily_aligned_shard",
    "inspect_first_rows_response",
    "inspect_gamma_market",
    "inspect_parquet_index",
    "inspect_repo_metadata",
    "inspect_repo_tree",
]
