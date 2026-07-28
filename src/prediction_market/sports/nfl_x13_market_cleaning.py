"""Auditable row-level market cleaning for the NFL X-13 factor lab.

The cleaner consumes already-normalized source observations together with the
raw row and explicit contract identity used to create them.  It never upgrades
an interval candle into a point quote, manufactures a missing Polymarket BBO,
or treats a derived complement as an observed/executable price.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from prediction_market import contracts
from prediction_market.sports.nfl_x13_capture import (
    BatchCaptureResult,
    GameCaptureResult,
    ImmutableRequestManifest,
    load_x13_capture,
)
from prediction_market.sports.nfl_x13_market import (
    MarketObservation,
    normalize_polymarket_trade,
)
from prediction_market.sports.nfl_x13_spec import (
    KALSHI_TEAM_ALIASES,
    canonical_sha256,
)
from prediction_market.sports.nfl_x13_universe import (
    UniverseContractV1,
    UniverseGameV1,
    X13UniverseBatchV1,
)

if TYPE_CHECKING:
    from prediction_market.sports.nfl_x13_pipeline import (
        VerifiedCaptureDocumentV1,
        _CurrentX13DatasetAuthorityV1,
    )


class MarketCleaningError(ValueError):
    """Cleaned observations do not have a safe stable identity."""


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise TypeError(f"{field} must be nonempty text or None")
    return value


def _canonical_orientation(value: str | None) -> str | None:
    if value is None:
        return None
    aliases = {native: canonical for canonical, native in KALSHI_TEAM_ALIASES}
    return aliases.get(value, value)


def _timestamp(value) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class MarketCleaningInputV1:
    """One normalized observation bound to its raw source identity."""

    game_id: str
    observation: MarketObservation
    contract_id: str | None
    event_id: str | None
    token_id: str | None
    ticker: str | None
    proposition: str | None
    orientation: str | None
    raw_row: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.game_id) is not str or not self.game_id.strip():
            raise TypeError("game_id must be nonempty text")
        if not isinstance(self.observation, MarketObservation):
            raise TypeError("observation must be MarketObservation")
        for field in (
            "contract_id",
            "event_id",
            "token_id",
            "ticker",
            "proposition",
            "orientation",
        ):
            _optional_text(getattr(self, field), field=field)
        if not isinstance(self.raw_row, Mapping):
            raise TypeError("raw_row must be a mapping")

    def to_dict(self) -> dict[str, object]:
        return {
            "game_id": self.game_id,
            "observation_id": self.observation.observation_id,
            "contract_id": self.contract_id,
            "event_id": self.event_id,
            "token_id": self.token_id,
            "ticker": self.ticker,
            "proposition": self.proposition,
            "orientation": self.orientation,
            "raw_row": dict(self.raw_row),
        }


def build_polymarket_cleaning_input_v1(
    raw_row: Mapping[str, object],
    *,
    game_id: str,
    raw_market_id: str,
    logical_market_id: str,
    outcome: str,
    contract_id: str | None,
    event_id: str | None,
    token_id: str | None,
    proposition: str | None,
    orientation: str | None,
) -> MarketCleaningInputV1:
    """Normalize a Polymarket trade while preserving absent native identity."""

    if not isinstance(raw_row, Mapping):
        raise TypeError("raw_row must be a mapping")
    normalized_raw = dict(raw_row)
    supplied_native_id = normalized_raw.get("transactionHash")
    native_missing = supplied_native_id is None or supplied_native_id == ""
    if native_missing:
        normalized_raw["transactionHash"] = canonical_sha256(
            {
                "contract_id": contract_id,
                "event_id": event_id,
                "logical_market_id": logical_market_id,
                "outcome": outcome,
                "raw_fingerprint": canonical_sha256(raw_row),
                "raw_market_id": raw_market_id,
                "schema_version": "polymarket_missing_native_provisional_v1",
                "token_id": token_id,
            }
        )
    observation = normalize_polymarket_trade(
        normalized_raw,
        raw_market_id=raw_market_id,
        logical_market_id=logical_market_id,
        outcome=outcome,
    )
    if native_missing:
        observation = replace(
            observation,
            observation_id=canonical_sha256(
                {
                    "logical_market_id": logical_market_id,
                    "outcome": outcome,
                    "raw_fingerprint": canonical_sha256(raw_row),
                    "schema_version": (
                        "polymarket_unresolved_native_observation_v1"
                    ),
                    "token_id": token_id,
                }
            ),
            native_id="",
        )
    return MarketCleaningInputV1(
        game_id=game_id,
        observation=observation,
        contract_id=contract_id,
        event_id=event_id,
        token_id=token_id,
        ticker=None,
        proposition=proposition,
        orientation=orientation,
        raw_row=raw_row,
    )


@dataclass(frozen=True, slots=True)
class MarketCleaningAuditRowV1:
    game_id: str
    audit_row_id: str
    observation_id: str
    contract_id: str | None
    event_id: str | None
    token_id: str | None
    ticker: str | None
    logical_market_id: str | None
    venue: str
    proposition: str | None
    orientation: str | None
    outcome: str
    observation_kind: Literal[
        "actual_trade",
        "kalshi_1m_candle",
        "derived_complement",
        "unknown",
    ]
    source_interval_start_utc: str
    source_interval_end_utc: str
    native_id: str | None
    dedupe_key: str
    dedupe_decision: Literal[
        "KEEP",
        "DROP_EXACT_DUPLICATE",
        "EXCLUDE_COLLISION",
        "EXCLUDE",
    ]
    collision_reason: str | None
    unknown_reason: str | None
    exclusion_reason: str | None
    provenance: str
    raw_fingerprint: str
    actual_observation: bool
    actual_trade: bool
    bbo_available: bool
    bbo_tick: bool
    executable: bool
    included: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "game_id": self.game_id,
            "audit_row_id": self.audit_row_id,
            "observation_id": self.observation_id,
            "contract_id": self.contract_id,
            "event_id": self.event_id,
            "token_id": self.token_id,
            "ticker": self.ticker,
            "logical_market_id": self.logical_market_id,
            "venue": self.venue,
            "proposition": self.proposition,
            "orientation": self.orientation,
            "outcome": self.outcome,
            "observation_kind": self.observation_kind,
            "source_interval_start_utc": self.source_interval_start_utc,
            "source_interval_end_utc": self.source_interval_end_utc,
            "native_id": self.native_id,
            "dedupe_key": self.dedupe_key,
            "dedupe_decision": self.dedupe_decision,
            "collision_reason": self.collision_reason,
            "unknown_reason": self.unknown_reason,
            "exclusion_reason": self.exclusion_reason,
            "provenance": self.provenance,
            "raw_fingerprint": self.raw_fingerprint,
            "actual_observation": self.actual_observation,
            "actual_trade": self.actual_trade,
            "bbo_available": self.bbo_available,
            "bbo_tick": self.bbo_tick,
            "executable": self.executable,
            "included": self.included,
        }


@dataclass(frozen=True, slots=True)
class MarketCleaningAuditV1:
    status: Literal["PASS", "FAIL"]
    rows: tuple[MarketCleaningAuditRowV1, ...]
    cleaned_observations: tuple[MarketObservation, ...]

    def to_records(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.rows]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "MarketCleaningAuditV1",
            "status": self.status,
            "input_row_count": len(self.rows),
            "included_row_count": sum(row.included for row in self.rows),
            "cleaned_observation_count": len(self.cleaned_observations),
            "rows": self.to_records(),
        }


def _observation_kind(
    observation: MarketObservation,
) -> Literal[
    "actual_trade",
    "kalshi_1m_candle",
    "derived_complement",
    "unknown",
]:
    if observation.provenance == "derived_complement":
        return "derived_complement"
    if observation.kind == "trade" and observation.provenance == "observed":
        return "actual_trade"
    if (
        observation.kind == "kalshi_1m_candle_bbo"
        and observation.venue == "kalshi"
        and observation.provenance == "observed"
    ):
        return "kalshi_1m_candle"
    return "unknown"


def _row_reasons(
    item: MarketCleaningInputV1,
    *,
    kind: str,
) -> tuple[str | None, str | None]:
    observation = item.observation
    if kind == "unknown":
        unknown_reason = "unknown_observation_kind"
    elif item.contract_id is None:
        unknown_reason = "missing_contract_id"
    elif item.event_id is None:
        unknown_reason = "missing_event_id"
    elif observation.venue == "polymarket" and item.token_id is None:
        unknown_reason = "missing_polymarket_token_id"
    elif observation.venue == "kalshi" and item.ticker is None:
        unknown_reason = "missing_kalshi_ticker"
    elif (
        observation.venue == "kalshi"
        and kind == "actual_trade"
        and not observation.native_id
    ):
        unknown_reason = "missing_kalshi_trade_id"
    elif item.proposition is None:
        unknown_reason = "missing_proposition"
    elif item.orientation is None:
        unknown_reason = "orientation_unknown"
    else:
        unknown_reason = None

    if item.orientation is None:
        exclusion_reason = "invalid_orientation"
    elif kind == "actual_trade" and (
        observation.bid is not None or observation.ask is not None
    ):
        exclusion_reason = "actual_trade_contains_bbo"
    elif kind == "kalshi_1m_candle" and observation.price is not None:
        exclusion_reason = "kalshi_candle_contains_point_price"
    elif kind == "derived_complement" and (
        observation.derived_from_observation_id is None
        or observation.bid is not None
        or observation.ask is not None
        or observation.primary_path_eligible
    ):
        exclusion_reason = "derived_complement_is_observed_or_executable"
    else:
        exclusion_reason = unknown_reason
    return unknown_reason, exclusion_reason


def _dedupe_key(
    item: MarketCleaningInputV1,
    *,
    raw_fingerprint: str,
) -> str:
    observation = item.observation
    if observation.native_id:
        return canonical_sha256(
            (
                item.game_id,
                observation.venue,
                observation.provenance,
                _observation_kind(observation),
                observation.outcome,
                "native_id",
                observation.native_id,
            )
        )
    return canonical_sha256(
        (
            item.game_id,
            observation.venue,
            observation.provenance,
            _observation_kind(observation),
            item.contract_id,
            item.event_id,
            item.token_id,
            item.ticker,
            observation.logical_market_id,
            observation.outcome,
            observation.source_interval.start,
            observation.source_interval.end,
            observation.price,
            observation.size,
            raw_fingerprint,
        )
    )


def _ohlc_tuple(value) -> tuple[object, ...] | None:
    if value is None:
        return None
    return (value.open, value.high, value.low, value.close)


def _observation_economic_tuple(
    observation: MarketObservation,
) -> tuple[object, ...]:
    return (
        observation.venue,
        observation.raw_market_id,
        observation.logical_market_id,
        observation.outcome,
        observation.kind,
        observation.provenance,
        observation.source_interval.start,
        observation.source_interval.end,
        observation.source_interval.end_inclusive,
        observation.price,
        observation.size,
        observation.bid,
        observation.ask,
        observation.is_block_trade,
        observation.taker_side,
        observation.taker_outcome_side,
        observation.yes_price_dollars,
        observation.no_price_dollars,
        observation.primary_path_eligible,
        _ohlc_tuple(observation.yes_bid_ohlc),
        _ohlc_tuple(observation.yes_ask_ohlc),
        observation.volume,
        observation.open_interest,
        observation.derived_from_observation_id,
    )


def _economic_tuple(
    item: MarketCleaningInputV1,
) -> tuple[object, ...]:
    return _observation_economic_tuple(item.observation)


def deduplicate_cleaned_observations_v1(
    observations: Sequence[MarketObservation],
) -> tuple[MarketObservation, ...]:
    """Deduplicate stable IDs and reject conflicting normalized economics."""

    if not isinstance(observations, Sequence) or isinstance(
        observations,
        (str, bytes),
    ):
        raise TypeError("observations must be a sequence")
    by_identity: dict[
        tuple[str, str, str, str, str],
        tuple[str, MarketObservation],
    ] = {}
    ordered: list[MarketObservation] = []
    for observation in observations:
        if not isinstance(observation, MarketObservation):
            raise TypeError("observations must contain MarketObservation")
        if not observation.native_id:
            raise MarketCleaningError(
                "cleaned observation is missing stable native/fallback identity"
            )
        identity = (
            observation.venue,
            observation.provenance,
            _observation_kind(observation),
            observation.outcome,
            observation.native_id,
        )
        economics = canonical_sha256(
            _observation_economic_tuple(observation)
        )
        previous = by_identity.get(identity)
        if previous is None:
            by_identity[identity] = (economics, observation)
            ordered.append(observation)
        elif previous[0] != economics:
            raise MarketCleaningError(
                "conflicting economics for stable observation identity"
            )
    return tuple(ordered)


def _semantic_identity(
    item: MarketCleaningInputV1,
) -> str:
    observation = item.observation
    kind = _observation_kind(observation)
    if observation.native_id:
        material: tuple[object, ...] = (
            item.game_id,
            observation.venue,
            kind,
            observation.provenance,
            observation.outcome,
            "native_id",
            observation.native_id,
        )
    else:
        material = (
            item.game_id,
            observation.venue,
            kind,
            observation.provenance,
            item.contract_id,
            item.event_id,
            item.token_id,
            item.ticker,
            observation.raw_market_id,
            observation.logical_market_id,
            observation.outcome,
            observation.source_interval.start,
            observation.source_interval.end,
            observation.source_interval.end_inclusive,
        )
    return canonical_sha256(material)


def _fallback_observation(
    observation: MarketObservation,
    *,
    dedupe_key: str,
    economic_tuple: tuple[object, ...],
) -> MarketObservation:
    if observation.native_id:
        return observation
    return replace(
        observation,
        observation_id=canonical_sha256(
            (
                "market_cleaning_observation_v1",
                dedupe_key,
                economic_tuple,
            )
        ),
        native_id=f"fallback:{dedupe_key.removeprefix('sha256:')}",
    )


def clean_market_rows_v1(
    inputs: Sequence[MarketCleaningInputV1],
) -> MarketCleaningAuditV1:
    """Create row-level market provenance without changing evidence kind."""

    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        raise TypeError("inputs must be a sequence")
    if any(not isinstance(item, MarketCleaningInputV1) for item in inputs):
        raise TypeError("inputs must contain MarketCleaningInputV1")
    if len({item.game_id for item in inputs}) > 1:
        raise MarketCleaningError(
            "market cleaning inputs must contain exactly one game_id"
        )

    rows: list[MarketCleaningAuditRowV1] = []
    candidates: list[MarketObservation] = []
    semantic_groups: dict[str, list[int]] = {}
    economic_hashes: list[str] = []
    for ordinal, item in enumerate(inputs):
        observation = item.observation
        raw_fingerprint = canonical_sha256(item.raw_row)
        kind = _observation_kind(observation)
        unknown_reason, exclusion_reason = _row_reasons(
            item,
            kind=kind,
        )
        included = unknown_reason is None and exclusion_reason is None
        dedupe_key = _dedupe_key(item, raw_fingerprint=raw_fingerprint)
        economics = _economic_tuple(item)
        candidate = _fallback_observation(
            observation,
            dedupe_key=dedupe_key,
            economic_tuple=economics,
        )
        row = MarketCleaningAuditRowV1(
            game_id=item.game_id,
            audit_row_id=canonical_sha256(
                {
                    "dedupe_key": dedupe_key,
                    "game_id": item.game_id,
                    "input_ordinal": ordinal,
                    "raw_fingerprint": raw_fingerprint,
                }
            ),
            observation_id=candidate.observation_id,
            contract_id=item.contract_id,
            event_id=item.event_id,
            token_id=item.token_id,
            ticker=item.ticker,
            logical_market_id=observation.logical_market_id,
            venue=observation.venue,
            proposition=item.proposition,
            orientation=_canonical_orientation(item.orientation),
            outcome=observation.outcome,
            observation_kind=kind,
            source_interval_start_utc=_timestamp(
                observation.source_interval.start
            ),
            source_interval_end_utc=_timestamp(observation.source_interval.end),
            native_id=observation.native_id or None,
            dedupe_key=dedupe_key,
            dedupe_decision="KEEP" if included else "EXCLUDE",
            collision_reason=None,
            unknown_reason=unknown_reason,
            exclusion_reason=exclusion_reason,
            provenance=observation.provenance,
            raw_fingerprint=raw_fingerprint,
            actual_observation=observation.provenance == "observed",
            actual_trade=kind == "actual_trade",
            bbo_available=kind == "kalshi_1m_candle"
            and observation.bid is not None
            and observation.ask is not None,
            bbo_tick=False,
            executable=False,
            included=included,
        )
        rows.append(row)
        candidates.append(candidate)
        economic_hashes.append(canonical_sha256(economics))
        semantic_groups.setdefault(_semantic_identity(item), []).append(ordinal)

    for indexes in semantic_groups.values():
        hashes = {economic_hashes[index] for index in indexes}
        if len(hashes) > 1:
            for index in indexes:
                rows[index] = replace(
                    rows[index],
                    dedupe_decision="EXCLUDE_COLLISION",
                    collision_reason=(
                        "conflicting_economics_for_semantic_identity"
                    ),
                    exclusion_reason=(
                        "conflicting_economics_for_semantic_identity"
                    ),
                    included=False,
                )
            continue
        seen_dedupe_keys: set[str] = set()
        for index in indexes:
            if not rows[index].included:
                continue
            if rows[index].dedupe_key in seen_dedupe_keys:
                rows[index] = replace(
                    rows[index],
                    dedupe_decision="DROP_EXACT_DUPLICATE",
                    exclusion_reason="exact_duplicate",
                    included=False,
                )
            else:
                seen_dedupe_keys.add(rows[index].dedupe_key)

    failed = any(
        row.dedupe_decision in {"EXCLUDE_COLLISION", "EXCLUDE"}
        for row in rows
    )
    status: Literal["PASS", "FAIL"] = "FAIL" if failed else "PASS"
    cleaned = tuple(
        candidates[index]
        for index, row in enumerate(rows)
        if row.dedupe_decision == "KEEP" and row.included
    )
    return MarketCleaningAuditV1(
        status=status,
        rows=tuple(rows),
        cleaned_observations=cleaned if status == "PASS" else (),
    )


class X13MarketRecleanError(ValueError):
    """Immutable raw evidence cannot support an exact V2 market row."""


def load_x13_universe_artifact(artifact: str | Path):
    """Lazy import avoids the pipeline -> association -> cleaner cycle."""

    from prediction_market.sports.nfl_x13_pipeline import (
        load_x13_universe_artifact as implementation,
    )

    return implementation(artifact)


@dataclass(frozen=True, slots=True)
class X13MarketCleaningAuditRowV2:
    """One normalized observation decision tied to one exact raw row."""

    game_id: str
    audit_row_id: str
    observation_id: str
    contract_id: str
    logical_market_id: str
    venue: str
    outcome: str
    observation_kind: str
    provenance: str
    source_interval_start_utc: str
    source_interval_end_utc: str
    native_id: str
    capture_sha256: str
    raw_manifest_sha256: str
    captured_license_status: str
    current_license_status: str
    current_authority_snapshot_sha256: str
    current_catalog_registry_sha256: str
    current_data_license_registry_sha256: str
    current_dataset_registry_sha256: str
    current_experiment_registry_sha256: str
    raw_object_sha256: str
    raw_request_sha256: str
    raw_row_index: int
    raw_row_fingerprint: str
    raw_source_url: str
    raw_source_cursor: str | None
    raw_page_index: int
    raw_cursor_in: str | None
    raw_cursor_out: str | None
    terminal_proof: str
    terminal_cursor: str | None
    terminal_manifest_sha256: str
    condition_id: str | None
    token_id: str | None
    ticker: str | None
    dedupe_decision: Literal["KEEP", "DROP_EXACT_DUPLICATE"]
    exclusion_reason: str | None
    actual_observation: bool
    actual_trade: bool
    bbo_available: bool
    bbo_tick: bool
    executable: bool
    included: bool
    raw_request_row_available: Literal[True]
    raw_token_or_ticker_binding_available: Literal[True]
    v2_cleaning_equivalent: Literal[True]
    stage_semantics: Literal["RAW_CAPTURE_V2_RECLEAN_EQUIVALENT"]
    data_gap: None

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class X13MarketRecleanGameV2:
    """Compact V2 outputs for one game after whole-capture verification."""

    status: Literal["PASS"]
    game_id: str
    capture_sha256: str
    contract_inventory: tuple[Mapping[str, object], ...]
    normalized_observations: tuple[MarketObservation, ...]
    normalized_observation_rows: tuple[Mapping[str, object], ...]
    audit_rows: tuple[X13MarketCleaningAuditRowV2, ...]

    def contract_inventory_records(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.contract_inventory]

    def normalized_observation_records(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.normalized_observation_rows]

    def market_cleaning_audit_records(self) -> list[dict[str, object]]:
        return [row.to_dict() for row in self.audit_rows]


@dataclass(frozen=True, slots=True)
class VerifiedX13MarketManifestV2:
    """Payload-free facts for one globally verified immutable raw object."""

    game_id: str | None
    receipt: ImmutableRequestManifest
    resource: str
    dataset_id: str
    source_version: str
    license_ref: str
    captured_license_status: str
    current_license_status: str


@dataclass(frozen=True, slots=True)
class VerifiedX13MarketCaptureIndexV2:
    """Immutable, thread-safe index over a fully hash-verified X-13 capture."""

    status: Literal["PASS"]
    capture_sha256: str
    capture_receipt_sha256: str
    raw_manifest_count: int
    raw_byte_length: int
    manifest_hashes_verified: Literal[True]
    current_authority_snapshot_sha256: str
    current_authority_component_sha256s: Mapping[str, str]
    universe: X13UniverseBatchV1
    capture: BatchCaptureResult
    manifests_by_game: Mapping[
        str,
        tuple[VerifiedX13MarketManifestV2, ...],
    ]
    batch_manifests: tuple[VerifiedX13MarketManifestV2, ...]
    raw_store_root: Path
    program_root: Path


@dataclass(frozen=True, slots=True)
class _RawLineagePageV2:
    document: VerifiedCaptureDocumentV1
    receipt: ImmutableRequestManifest
    page_index: int
    cursor_in: str | None
    cursor_out: str | None
    terminal_proof: str
    terminal_cursor: str | None
    terminal_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _RawObservationSeedV2:
    observation: MarketObservation
    contract: UniverseContractV1
    raw_row: Mapping[str, object]
    raw_row_index: int
    page: _RawLineagePageV2
    condition_id: str | None
    token_id: str | None
    ticker: str | None


_V2_CAPTURE_RESOURCES = frozenset(
    {
        "gamma_event",
        "kalshi_candlesticks",
        "kalshi_markets",
        "kalshi_trades",
        "polymarket_trades",
    }
)
_V2_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _require_v2_sha256(value: object, field: str) -> str:
    if type(value) is not str or _V2_SHA256_RE.fullmatch(value) is None:
        raise X13MarketRecleanError(f"{field} is not a canonical SHA-256")
    return value


def _v2_timestamp(observation: MarketObservation, *, end: bool) -> str:
    value = (
        observation.source_interval.end
        if end
        else observation.source_interval.start
    )
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _document_params(
    document: VerifiedCaptureDocumentV1,
) -> Mapping[str, object]:
    request = document.source_request
    if not isinstance(request, Mapping):
        raise X13MarketRecleanError("verified source request is malformed")
    params = request.get("params")
    if not isinstance(params, Mapping):
        raise X13MarketRecleanError(
            "verified source request params are malformed"
        )
    return params


def _validate_verified_game_inputs_v2(
    *,
    game: UniverseGameV1,
    capture_game: GameCaptureResult,
    documents: Sequence[VerifiedCaptureDocumentV1],
    capture_sha256: str,
) -> tuple[
    tuple[VerifiedCaptureDocumentV1, ...],
    Mapping[str, ImmutableRequestManifest],
]:
    from prediction_market.sports.nfl_x13_pipeline import (
        VerifiedCaptureDocumentV1,
    )

    _require_v2_sha256(capture_sha256, "capture_sha256")
    if not isinstance(game, UniverseGameV1):
        raise X13MarketRecleanError("game has an unknown type")
    if not isinstance(capture_game, GameCaptureResult):
        raise X13MarketRecleanError("capture_game has an unknown type")
    if capture_game.game_id != game.game_id:
        raise X13MarketRecleanError(
            "capture game differs from universe game"
        )
    if (
        not isinstance(documents, Sequence)
        or isinstance(documents, (str, bytes))
        or any(
            not isinstance(document, VerifiedCaptureDocumentV1)
            for document in documents
        )
    ):
        raise X13MarketRecleanError(
            "documents must contain verified capture documents"
        )
    values = tuple(documents)
    by_manifest: dict[str, VerifiedCaptureDocumentV1] = {}
    authority_bindings: set[tuple[str, str, str, str, str]] = set()
    for document in values:
        if document.game_id != game.game_id:
            raise X13MarketRecleanError(
                "verified document belongs to another game"
            )
        if document.resource not in _V2_CAPTURE_RESOURCES:
            raise X13MarketRecleanError(
                f"unknown verified market resource: {document.resource}"
            )
        if document.license_status not in {
            "approved",
            "research_only",
            "pending",
        }:
            raise X13MarketRecleanError(
                "captured license status is unknown or blocked"
            )
        if document.current_license_status not in {
            "approved",
            "research_only",
        }:
            raise X13MarketRecleanError(
                "current dataset license is not eligible"
            )
        authority_bindings.add(
            (
                _require_v2_sha256(
                    document.current_authority_snapshot_sha256,
                    "current authority snapshot SHA-256",
                ),
                _require_v2_sha256(
                    document.current_catalog_registry_sha256,
                    "current catalog registry SHA-256",
                ),
                _require_v2_sha256(
                    document.current_data_license_registry_sha256,
                    "current data license registry SHA-256",
                ),
                _require_v2_sha256(
                    document.current_dataset_registry_sha256,
                    "current dataset registry SHA-256",
                ),
                _require_v2_sha256(
                    document.current_experiment_registry_sha256,
                    "current experiment registry SHA-256",
                ),
            )
        )
        if document.manifest_sha256 in by_manifest:
            raise X13MarketRecleanError(
                "verified document manifest identity collided"
            )
        by_manifest[document.manifest_sha256] = document
    if len(authority_bindings) != 1:
        raise X13MarketRecleanError(
            "verified documents do not bind one current authority snapshot"
        )
    receipts: dict[str, ImmutableRequestManifest] = {}
    for receipt in capture_game.manifests:
        if not isinstance(receipt, ImmutableRequestManifest):
            raise X13MarketRecleanError(
                "capture game contains an unknown manifest receipt"
            )
        if receipt.manifest_sha256 in receipts:
            raise X13MarketRecleanError(
                "capture manifest receipt identity collided"
            )
        receipts[receipt.manifest_sha256] = receipt
    if set(receipts) != set(by_manifest):
        raise X13MarketRecleanError(
            "verified documents differ from capture game manifests"
        )
    for manifest_sha256, document in by_manifest.items():
        receipt = receipts[manifest_sha256]
        if (
            receipt.object_sha256 != document.object_sha256
            or receipt.byte_length != document.byte_length
        ):
            raise X13MarketRecleanError(
                "verified document differs from immutable receipt"
            )
    return values, MappingProxyType(receipts)


@dataclass(frozen=True, slots=True)
class _ObservationIndexV2:
    observed_by_key: Mapping[tuple[object, ...], MarketObservation]
    children_by_parent: Mapping[str, tuple[MarketObservation, ...]]


def _observation_lookup_key_v2(
    *,
    venue: str,
    raw_market_id: str,
    logical_market_id: str | None,
    outcome: str,
    native_id: str,
    kind: str,
) -> tuple[object, ...]:
    return (
        venue,
        raw_market_id,
        logical_market_id,
        outcome,
        native_id,
        kind,
    )


def _build_observation_index_v2(
    observations: Sequence[MarketObservation],
) -> _ObservationIndexV2:
    observed_by_key: dict[tuple[object, ...], MarketObservation] = {}
    children_by_parent: dict[str, list[MarketObservation]] = {}
    observation_ids: set[str] = set()
    observed_ids: set[str] = set()
    for observation in observations:
        if not isinstance(observation, MarketObservation):
            raise X13MarketRecleanError(
                "normalized observation index has an unknown row"
            )
        if observation.observation_id in observation_ids:
            raise X13MarketRecleanError(
                "normalized observation identity collided"
            )
        observation_ids.add(observation.observation_id)
        if observation.provenance == "observed":
            observed_ids.add(observation.observation_id)
            key = _observation_lookup_key_v2(
                venue=observation.venue,
                raw_market_id=observation.raw_market_id,
                logical_market_id=observation.logical_market_id,
                outcome=observation.outcome,
                native_id=observation.native_id,
                kind=observation.kind,
            )
            if key in observed_by_key:
                raise X13MarketRecleanError(
                    "stable raw identity collided across "
                    "normalized observations"
                )
            observed_by_key[key] = observation
        else:
            parent = observation.derived_from_observation_id
            if type(parent) is not str or not parent:
                raise X13MarketRecleanError(
                    "derived observation has no canonical parent"
                )
            children_by_parent.setdefault(parent, []).append(observation)
    for parent, children in children_by_parent.items():
        if parent not in observed_ids or len(children) != 1:
            raise X13MarketRecleanError(
                "normalized observation has invalid derived complements"
            )
    return _ObservationIndexV2(
        observed_by_key=MappingProxyType(observed_by_key),
        children_by_parent=MappingProxyType(
            {
                parent: tuple(children)
                for parent, children in children_by_parent.items()
            }
        ),
    )


def _indexed_observed_match_v2(
    index: _ObservationIndexV2,
    *,
    venue: str,
    raw_market_id: str,
    logical_market_id: str,
    outcome: str,
    native_id: object,
    kind: str,
) -> MarketObservation:
    if type(native_id) is not str or not native_id:
        raise X13MarketRecleanError(
            "raw row does not bind exactly one normalized observation"
        )
    observation = index.observed_by_key.get(
        _observation_lookup_key_v2(
            venue=venue,
            raw_market_id=raw_market_id,
            logical_market_id=logical_market_id,
            outcome=outcome,
            native_id=native_id,
            kind=kind,
        )
    )
    if observation is None:
        raise X13MarketRecleanError(
            "raw row does not bind exactly one normalized observation"
        )
    return observation


def _observation_and_children_v2(
    index: _ObservationIndexV2,
    observed: MarketObservation,
) -> tuple[MarketObservation, ...]:
    children = index.children_by_parent.get(observed.observation_id, ())
    return (observed, *children)


def _polymarket_pages_v2(
    capture_game: GameCaptureResult,
    documents: Sequence[VerifiedCaptureDocumentV1],
    receipts: Mapping[str, ImmutableRequestManifest],
) -> tuple[_RawLineagePageV2, ...]:
    document_by_manifest = {
        document.manifest_sha256: document for document in documents
    }
    grouped: dict[str, list[object]] = {}
    seen_manifests: set[str] = set()
    for proof in capture_game.polymarket_terminal_windows:
        if proof.manifest_sha256 in seen_manifests:
            raise X13MarketRecleanError(
                "Polymarket terminal manifest identity collided"
            )
        seen_manifests.add(proof.manifest_sha256)
        if proof.manifest_sha256 not in document_by_manifest:
            raise X13MarketRecleanError(
                "Polymarket terminal proof has no verified document"
            )
        grouped.setdefault(proof.condition_id, []).append(proof)
    values: list[_RawLineagePageV2] = []
    for condition_id in sorted(grouped):
        proofs = sorted(
            grouped[condition_id],
            key=lambda proof: (
                proof.start_ts,
                proof.end_ts,
                proof.manifest_sha256,
            ),
        )
        for page_index, proof in enumerate(proofs):
            document = document_by_manifest[proof.manifest_sha256]
            if document.resource != "polymarket_trades":
                raise X13MarketRecleanError(
                    "Polymarket terminal proof resource is unknown"
                )
            params = _document_params(document)
            if (
                params.get("market") != condition_id
                or params.get("start") != proof.start_ts
                or params.get("end") != proof.end_ts
                or type(document.payload) is not list
                or len(document.payload) != proof.item_count
            ):
                raise X13MarketRecleanError(
                    "Polymarket terminal proof differs from raw request"
                )
            offset = params.get("offset")
            if type(offset) is not int or offset < 0:
                raise X13MarketRecleanError(
                    "Polymarket raw request offset is unknown"
                )
            values.append(
                _RawLineagePageV2(
                    document=document,
                    receipt=receipts[document.manifest_sha256],
                    page_index=page_index,
                    cursor_in=str(offset),
                    cursor_out=None,
                    terminal_proof=proof.terminal_proof,
                    terminal_cursor=None,
                    terminal_manifest_sha256=document.manifest_sha256,
                )
            )
    return tuple(values)


def _kalshi_trade_pages_v2(
    capture_game: GameCaptureResult,
    documents: Sequence[VerifiedCaptureDocumentV1],
    receipts: Mapping[str, ImmutableRequestManifest],
) -> Mapping[str, tuple[_RawLineagePageV2, ...]]:
    proofs: dict[str, object] = {}
    for proof in capture_game.kalshi_trade_proofs:
        if proof.ticker in proofs:
            raise X13MarketRecleanError(
                "Kalshi trade proof ticker identity collided"
            )
        proofs[proof.ticker] = proof
    documents_by_ticker: dict[
        str,
        list[VerifiedCaptureDocumentV1],
    ] = {}
    for document in documents:
        if document.resource != "kalshi_trades":
            continue
        ticker = _document_params(document).get("ticker")
        if type(ticker) is not str or ticker not in proofs:
            raise X13MarketRecleanError(
                "Kalshi trade document has no exact ticker proof"
            )
        documents_by_ticker.setdefault(ticker, []).append(document)
    if set(documents_by_ticker) != set(proofs):
        raise X13MarketRecleanError(
            "Kalshi trade proofs and verified pages differ"
        )

    result: dict[str, tuple[_RawLineagePageV2, ...]] = {}
    for ticker in sorted(proofs):
        proof = proofs[ticker]
        by_cursor: dict[str | None, VerifiedCaptureDocumentV1] = {}
        for document in documents_by_ticker[ticker]:
            params = _document_params(document)
            raw_cursor = params.get("cursor")
            if raw_cursor is None:
                cursor_in = None
            elif type(raw_cursor) is str and raw_cursor:
                cursor_in = raw_cursor
            else:
                raise X13MarketRecleanError(
                    "Kalshi request cursor is malformed"
                )
            if cursor_in in by_cursor:
                raise X13MarketRecleanError(
                    "Kalshi request cursor identity collided"
                )
            by_cursor[cursor_in] = document
        ordered: list[
            tuple[VerifiedCaptureDocumentV1, str | None, str]
        ] = []
        cursor: str | None = None
        while True:
            document = by_cursor.get(cursor)
            if document is None:
                raise X13MarketRecleanError(
                    "Kalshi cursor chain is incomplete"
                )
            payload = document.payload
            if (
                type(payload) is not dict
                or type(payload.get("trades")) is not list
                or type(payload.get("cursor")) is not str
            ):
                raise X13MarketRecleanError(
                    "Kalshi trade page is malformed"
                )
            cursor_out = payload["cursor"]
            ordered.append((document, cursor, cursor_out))
            if cursor_out == "":
                break
            if cursor_out in {value[1] for value in ordered}:
                raise X13MarketRecleanError(
                    "Kalshi cursor chain collided"
                )
            cursor = cursor_out
        if (
            len(ordered) != len(by_cursor)
            or len(ordered) != proof.page_count
            or proof.terminal_proof != "explicit_empty_cursor"
            or proof.terminal_cursor != ""
        ):
            raise X13MarketRecleanError(
                "Kalshi terminal cursor proof differs from verified pages"
            )
        terminal_manifest = ordered[-1][0].manifest_sha256
        result[ticker] = tuple(
            _RawLineagePageV2(
                document=document,
                receipt=receipts[document.manifest_sha256],
                page_index=index,
                cursor_in=cursor_in,
                cursor_out=cursor_out,
                terminal_proof=proof.terminal_proof,
                terminal_cursor=proof.terminal_cursor,
                terminal_manifest_sha256=terminal_manifest,
            )
            for index, (document, cursor_in, cursor_out) in enumerate(
                ordered
            )
        )
    return MappingProxyType(result)


def _kalshi_candle_pages_v2(
    capture_game: GameCaptureResult,
    documents: Sequence[VerifiedCaptureDocumentV1],
    receipts: Mapping[str, ImmutableRequestManifest],
) -> tuple[_RawLineagePageV2, ...]:
    proof_by_manifest: dict[str, object] = {}
    for proof in capture_game.kalshi_candlesticks:
        if proof.manifest_sha256 in proof_by_manifest:
            raise X13MarketRecleanError(
                "Kalshi candlestick manifest identity collided"
            )
        proof_by_manifest[proof.manifest_sha256] = proof
    candle_documents = tuple(
        document
        for document in documents
        if document.resource == "kalshi_candlesticks"
    )
    if {
        document.manifest_sha256 for document in candle_documents
    } != set(proof_by_manifest):
        raise X13MarketRecleanError(
            "Kalshi candlestick proofs and verified pages differ"
        )
    values: list[_RawLineagePageV2] = []
    for document in sorted(
        candle_documents,
        key=lambda value: value.manifest_sha256,
    ):
        proof = proof_by_manifest[document.manifest_sha256]
        payload = document.payload
        if (
            type(payload) is not dict
            or payload.get("ticker") != proof.ticker
            or type(payload.get("candlesticks")) is not list
            or len(payload["candlesticks"]) != proof.candlestick_count
            or _document_params(document).get("period_interval")
            != proof.period_interval
        ):
            raise X13MarketRecleanError(
                "Kalshi candlestick proof differs from verified raw"
            )
        values.append(
            _RawLineagePageV2(
                document=document,
                receipt=receipts[document.manifest_sha256],
                page_index=0,
                cursor_in=None,
                cursor_out=None,
                terminal_proof=(
                    "single_manifest_candlestick_count_and_interval_bounds"
                ),
                terminal_cursor=None,
                terminal_manifest_sha256=document.manifest_sha256,
            )
        )
    return tuple(values)


def _raw_observation_seeds_v2(
    *,
    game: UniverseGameV1,
    capture_game: GameCaptureResult,
    documents: Sequence[VerifiedCaptureDocumentV1],
    receipts: Mapping[str, ImmutableRequestManifest],
    observations: Sequence[MarketObservation],
) -> tuple[_RawObservationSeedV2, ...]:
    from prediction_market.sports.nfl_x13_pipeline import (
        _canonical_market_outcome,
        _kalshi_yes_outcome,
        _raw_contract_map,
    )

    raw_contracts = _raw_contract_map(game)
    observation_index = _build_observation_index_v2(observations)
    token_bindings: dict[str, tuple[str, str]] = {}
    for binding in game.polymarket_token_bindings:
        for outcome, token_id in zip(
            binding.outcomes,
            binding.token_ids,
            strict=True,
        ):
            prior = token_bindings.get(token_id)
            value = (binding.condition_id, outcome)
            if prior is not None and prior != value:
                raise X13MarketRecleanError(
                    "Polymarket token identity collided"
                )
            token_bindings[token_id] = value

    seeds: list[_RawObservationSeedV2] = []
    for page in _polymarket_pages_v2(
        capture_game,
        documents,
        receipts,
    ):
        assert isinstance(page.document.payload, list)
        for raw_row_index, raw_value in enumerate(page.document.payload):
            if type(raw_value) is not dict:
                raise X13MarketRecleanError(
                    "Polymarket raw request row is malformed"
                )
            raw_row = raw_value
            token_id = raw_row.get("asset")
            if type(token_id) is not str or not token_id:
                raise X13MarketRecleanError(
                    "Polymarket token binding is missing"
                )
            token_binding = token_bindings.get(token_id)
            if (
                token_binding is None
                or raw_row.get("conditionId") != token_binding[0]
                or raw_row.get("outcome") != token_binding[1]
            ):
                raise X13MarketRecleanError(
                    "Polymarket token/outcome orientation is unresolved"
                )
            contract = raw_contracts.get(token_binding[0])
            if contract is None:
                raise X13MarketRecleanError(
                    "Polymarket raw row references an unknown contract"
                )
            outcome = _canonical_market_outcome(
                game,
                contract,
                token_binding[1],
            )
            observed = _indexed_observed_match_v2(
                observation_index,
                venue="polymarket",
                raw_market_id=contract.contract_id,
                logical_market_id=contract.logical_market_id,
                outcome=outcome,
                native_id=raw_row.get("transactionHash"),
                kind="trade",
            )
            seeds.extend(
                _RawObservationSeedV2(
                    observation=observation,
                    contract=contract,
                    raw_row=raw_row,
                    raw_row_index=raw_row_index,
                    page=page,
                    condition_id=token_binding[0],
                    token_id=token_id,
                    ticker=None,
                )
                for observation in _observation_and_children_v2(
                    observation_index,
                    observed,
                )
            )

    for ticker, pages in _kalshi_trade_pages_v2(
        capture_game,
        documents,
        receipts,
    ).items():
        contract = raw_contracts.get(ticker)
        if contract is None:
            raise X13MarketRecleanError(
                "Kalshi ticker references an unknown contract"
            )
        outcome = _canonical_market_outcome(
            game,
            contract,
            _kalshi_yes_outcome(
                contract,
                ticker=ticker,
                game_id=game.game_id,
            ),
        )
        for page in pages:
            payload = page.document.payload
            assert isinstance(payload, dict)
            trades = payload["trades"]
            assert isinstance(trades, list)
            for raw_row_index, raw_value in enumerate(trades):
                if type(raw_value) is not dict:
                    raise X13MarketRecleanError(
                        "Kalshi raw trade row is malformed"
                    )
                raw_row = raw_value
                if raw_row.get("ticker") != ticker:
                    raise X13MarketRecleanError(
                        "Kalshi raw trade ticker binding is unresolved"
                    )
                observed = _indexed_observed_match_v2(
                    observation_index,
                    venue="kalshi",
                    raw_market_id=ticker,
                    logical_market_id=contract.logical_market_id,
                    outcome=outcome,
                    native_id=raw_row.get("trade_id"),
                    kind="trade",
                )
                seeds.extend(
                    _RawObservationSeedV2(
                        observation=observation,
                        contract=contract,
                        raw_row=raw_row,
                        raw_row_index=raw_row_index,
                        page=page,
                        condition_id=None,
                        token_id=None,
                        ticker=ticker,
                    )
                    for observation in _observation_and_children_v2(
                        observation_index,
                        observed,
                    )
                )

    for page in _kalshi_candle_pages_v2(
        capture_game,
        documents,
        receipts,
    ):
        payload = page.document.payload
        assert isinstance(payload, dict)
        ticker = payload["ticker"]
        assert isinstance(ticker, str)
        contract = raw_contracts.get(ticker)
        if contract is None:
            raise X13MarketRecleanError(
                "Kalshi candle ticker references an unknown contract"
            )
        outcome = _canonical_market_outcome(
            game,
            contract,
            _kalshi_yes_outcome(
                contract,
                ticker=ticker,
                game_id=game.game_id,
            ),
        )
        candles = payload["candlesticks"]
        assert isinstance(candles, list)
        for raw_row_index, raw_value in enumerate(candles):
            if type(raw_value) is not dict:
                raise X13MarketRecleanError(
                    "Kalshi raw candle row is malformed"
                )
            raw_row = raw_value
            end_period_ts = raw_row.get("end_period_ts")
            if type(end_period_ts) is not int:
                raise X13MarketRecleanError(
                    "Kalshi raw candle time binding is unresolved"
                )
            observed = _indexed_observed_match_v2(
                observation_index,
                venue="kalshi",
                raw_market_id=ticker,
                logical_market_id=contract.logical_market_id,
                outcome=outcome,
                native_id=f"{ticker}:{end_period_ts}:1m",
                kind="kalshi_1m_candle_bbo",
            )
            seeds.append(
                _RawObservationSeedV2(
                    observation=observed,
                    contract=contract,
                    raw_row=raw_row,
                    raw_row_index=raw_row_index,
                    page=page,
                    condition_id=None,
                    token_id=None,
                    ticker=ticker,
                )
            )
    return tuple(seeds)


def _v2_audit_rows(
    *,
    game_id: str,
    capture_sha256: str,
    observations: Sequence[MarketObservation],
    seeds: Sequence[_RawObservationSeedV2],
) -> tuple[X13MarketCleaningAuditRowV2, ...]:
    seen: dict[str, str] = {}
    rows: list[X13MarketCleaningAuditRowV2] = []
    for seed in seeds:
        observation = seed.observation
        raw_fingerprint = canonical_sha256(seed.raw_row)
        prior = seen.get(observation.observation_id)
        if prior is None:
            dedupe_decision: Literal[
                "KEEP",
                "DROP_EXACT_DUPLICATE",
            ] = "KEEP"
            included = True
            exclusion_reason = None
            seen[observation.observation_id] = raw_fingerprint
        elif prior == raw_fingerprint:
            dedupe_decision = "DROP_EXACT_DUPLICATE"
            included = False
            exclusion_reason = "exact_duplicate"
        else:
            raise X13MarketRecleanError(
                "raw row fingerprints collided for normalized identity"
            )
        page = seed.page
        row = X13MarketCleaningAuditRowV2(
            game_id=game_id,
            audit_row_id=canonical_sha256(
                {
                    "capture_sha256": capture_sha256,
                    "game_id": game_id,
                    "manifest_sha256": page.document.manifest_sha256,
                    "observation_id": observation.observation_id,
                    "raw_row_index": seed.raw_row_index,
                    "schema_version": "X13MarketCleaningAuditRowV2",
                }
            ),
            observation_id=observation.observation_id,
            contract_id=seed.contract.contract_id,
            logical_market_id=seed.contract.logical_market_id,
            venue=observation.venue,
            outcome=observation.outcome,
            observation_kind=observation.kind,
            provenance=observation.provenance,
            source_interval_start_utc=_v2_timestamp(
                observation,
                end=False,
            ),
            source_interval_end_utc=_v2_timestamp(
                observation,
                end=True,
            ),
            native_id=observation.native_id,
            capture_sha256=capture_sha256,
            raw_manifest_sha256=page.document.manifest_sha256,
            captured_license_status=page.document.license_status,
            current_license_status=page.document.current_license_status,
            current_authority_snapshot_sha256=(
                page.document.current_authority_snapshot_sha256
            ),
            current_catalog_registry_sha256=(
                page.document.current_catalog_registry_sha256
            ),
            current_data_license_registry_sha256=(
                page.document.current_data_license_registry_sha256
            ),
            current_dataset_registry_sha256=(
                page.document.current_dataset_registry_sha256
            ),
            current_experiment_registry_sha256=(
                page.document.current_experiment_registry_sha256
            ),
            raw_object_sha256=page.document.object_sha256,
            raw_request_sha256=page.receipt.request_sha256,
            raw_row_index=seed.raw_row_index,
            raw_row_fingerprint=raw_fingerprint,
            raw_source_url=page.document.source_url,
            raw_source_cursor=page.document.source_cursor,
            raw_page_index=page.page_index,
            raw_cursor_in=page.cursor_in,
            raw_cursor_out=page.cursor_out,
            terminal_proof=page.terminal_proof,
            terminal_cursor=page.terminal_cursor,
            terminal_manifest_sha256=page.terminal_manifest_sha256,
            condition_id=seed.condition_id,
            token_id=seed.token_id,
            ticker=seed.ticker,
            dedupe_decision=dedupe_decision,
            exclusion_reason=exclusion_reason,
            actual_observation=observation.provenance == "observed",
            actual_trade=(
                observation.provenance == "observed"
                and observation.kind == "trade"
            ),
            bbo_available=(
                observation.bid is not None
                and observation.ask is not None
            ),
            bbo_tick=False,
            executable=False,
            included=included,
            raw_request_row_available=True,
            raw_token_or_ticker_binding_available=True,
            v2_cleaning_equivalent=True,
            stage_semantics="RAW_CAPTURE_V2_RECLEAN_EQUIVALENT",
            data_gap=None,
        )
        rows.append(row)
    expected = {observation.observation_id for observation in observations}
    kept = {
        row.observation_id
        for row in rows
        if row.dedupe_decision == "KEEP"
    }
    if kept != expected or sum(row.included for row in rows) != len(expected):
        raise X13MarketRecleanError(
            "normalized observations lack exact raw-row lineage"
        )
    return tuple(rows)


def _reclean_x13_market_game_documents_v2(
    *,
    game: UniverseGameV1,
    capture_game: GameCaptureResult,
    documents: Sequence[VerifiedCaptureDocumentV1],
    capture_sha256: str,
) -> X13MarketRecleanGameV2:
    """Reclean one game from selected, verified, parsed raw documents."""

    from prediction_market.sports.nfl_x13_pipeline import (
        X13PipelineError,
        _contract_payload,
        _normalize_observations,
        _observation_contracts,
        _observation_payload,
        normalize_x13_contracts,
    )

    values, receipts = _validate_verified_game_inputs_v2(
        game=game,
        capture_game=capture_game,
        documents=documents,
        capture_sha256=capture_sha256,
    )
    try:
        normalized_contracts = normalize_x13_contracts(game)
        observations = _normalize_observations(
            game,
            capture_game,
            values,
        )
        observation_contracts = _observation_contracts(
            observations,
            game,
        )
        seeds = _raw_observation_seeds_v2(
            game=game,
            capture_game=capture_game,
            documents=values,
            receipts=receipts,
            observations=observations,
        )
    except X13MarketRecleanError:
        raise
    except (X13PipelineError, KeyError, StopIteration) as error:
        raise X13MarketRecleanError(str(error)) from error

    audit_rows = _v2_audit_rows(
        game_id=game.game_id,
        capture_sha256=capture_sha256,
        observations=observations,
        seeds=seeds,
    )
    observations_by_logical: dict[str, list[MarketObservation]] = {}
    for observation in observations:
        contract = observation_contracts[observation.observation_id]
        observations_by_logical.setdefault(
            contract.logical_market_id,
            [],
        ).append(observation)
    normalized_contract_by_id = {
        contract.contract_id: contract
        for contract in normalized_contracts
    }
    contract_inventory = tuple(
        MappingProxyType(
            {
                "game_id": game.game_id,
                **_contract_payload(
                    contract,
                    normalized_contract_by_id[contract.contract_id],
                    observations_by_logical.get(
                        contract.logical_market_id,
                        (),
                    ),
                ),
            }
        )
        for contract in sorted(
            game.contracts,
            key=lambda value: value.contract_id,
        )
    )
    normalized_rows = tuple(
        MappingProxyType(
            {
                "game_id": game.game_id,
                "contract_id": observation_contracts[
                    observation.observation_id
                ].contract_id,
                **_observation_payload(
                    observation,
                    observation_contracts[observation.observation_id],
                ),
                "stage_semantics": "RAW_CAPTURE_V2_RECLEAN_EQUIVALENT",
                "historical_l2_available": False,
                "local_receive_time_available": False,
            }
        )
        for observation in observations
    )
    return X13MarketRecleanGameV2(
        status="PASS",
        game_id=game.game_id,
        capture_sha256=capture_sha256,
        contract_inventory=contract_inventory,
        normalized_observations=tuple(observations),
        normalized_observation_rows=normalized_rows,
        audit_rows=audit_rows,
    )


def _read_x13_market_manifest_index_entry_v2(
    receipt: ImmutableRequestManifest,
    *,
    game_id: str | None,
    current_authority: _CurrentX13DatasetAuthorityV1,
    raw_store_root: Path,
) -> VerifiedX13MarketManifestV2:
    """Read only immutable manifest metadata after the global object pass."""

    from prediction_market.sports.nfl_x13_pipeline import (
        _require_current_x13_dataset_authority,
        _resource,
        _strict_json,
    )

    if not isinstance(receipt, ImmutableRequestManifest):
        raise X13MarketRecleanError(
            "capture receipt has an unknown manifest type"
        )
    path = receipt.manifest_path
    try:
        relative = path.relative_to(raw_store_root)
        before = path.resolve(strict=True)
        before.relative_to(raw_store_root)
        if path.is_symlink() or not path.is_file():
            raise X13MarketRecleanError(
                "capture manifest path is missing or unsafe"
            )
        raw = path.read_bytes()
        after = path.resolve(strict=True)
        after.relative_to(raw_store_root)
    except (OSError, ValueError) as error:
        if isinstance(error, X13MarketRecleanError):
            raise
        raise X13MarketRecleanError(
            "capture manifest path is missing or unsafe"
        ) from error
    if before != after:
        raise X13MarketRecleanError(
            "capture manifest changed during index construction"
        )
    try:
        value = _strict_json(
            raw,
            context=f"capture manifest {receipt.manifest_sha256}",
        )
        if (
            type(value) is not dict
            or raw != contracts.canonical_json_bytes(value) + b"\n"
        ):
            raise X13MarketRecleanError(
                "capture manifest is not canonical JSON"
            )
        manifest = contracts.validate_static_dataset_manifest_intrinsic_v0(
            value
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, X13MarketRecleanError):
            raise
        raise X13MarketRecleanError(
            "capture manifest intrinsic validation failed"
        ) from error
    if (
        len(relative.parts) != 6
        or relative.parts[0] != "manifests"
        or not relative.parts[3].startswith("version=")
    ):
        raise X13MarketRecleanError(
            "capture manifest path is not canonical"
        )
    source_version = relative.parts[3].removeprefix("version=")
    expected_object_path = raw_store_root.joinpath(
        *Path(manifest.native_object_path).parts
    )
    fetched_at = (
        receipt.response_received_at_utc.isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    if (
        manifest.manifest_sha256 != receipt.manifest_sha256
        or manifest.object_sha256 != receipt.object_sha256
        or manifest.byte_length != receipt.byte_length
        or expected_object_path != receipt.object_path
        or manifest.upstream_partition
        != f"x13-{receipt.request_sha256.removeprefix('sha256:')}"
        or manifest.fetched_at != fetched_at
        or manifest.license_status in {"unknown", "blocked"}
    ):
        raise X13MarketRecleanError(
            "capture receipt does not bind immutable manifest metadata"
        )
    current = _require_current_x13_dataset_authority(
        current_authority,
        dataset_id=manifest.dataset_id,
        license_ref=manifest.license_ref,
        source_version=source_version,
    )
    return VerifiedX13MarketManifestV2(
        game_id=game_id,
        receipt=receipt,
        resource=_resource(manifest.source_url),
        dataset_id=manifest.dataset_id,
        source_version=source_version,
        license_ref=manifest.license_ref,
        captured_license_status=manifest.license_status,
        current_license_status=current.license_status,
    )


def _build_verified_x13_market_capture_index_v2(
    capture: BatchCaptureResult,
    *,
    universe: X13UniverseBatchV1,
    raw_store_root: Path,
    program_root: Path,
) -> VerifiedX13MarketCaptureIndexV2:
    """Bind one full intrinsic verification pass into a payload-free index."""

    from prediction_market.sports.nfl_x13_pipeline import (
        _load_current_x13_dataset_authority,
        _verify_pointer,
    )

    if not isinstance(capture, BatchCaptureResult):
        raise X13MarketRecleanError("capture has an unknown type")
    if not isinstance(universe, X13UniverseBatchV1):
        raise X13MarketRecleanError("universe has an unknown type")
    _verify_pointer(
        capture,
        universe=universe,
        program_root=program_root,
        raw_store_root=raw_store_root,
    )
    universe_game_ids = tuple(game.game_id for game in universe.games)
    capture_game_ids = tuple(game.game_id for game in capture.games)
    if capture_game_ids != universe_game_ids:
        raise X13MarketRecleanError(
            "capture game set differs from frozen universe"
        )
    current_authority = _load_current_x13_dataset_authority(program_root)
    components = MappingProxyType(
        {
            path: _require_v2_sha256(digest, f"{path} SHA-256")
            for path, digest in current_authority.component_sha256s.items()
        }
    )
    required_components = {
        "charter/catalog_registry.csv",
        "registries/data_license_register.csv",
        "registries/dataset_registry.csv",
        "registries/experiment_registry.csv",
    }
    if set(components) != required_components:
        raise X13MarketRecleanError(
            "current authority component set is incomplete"
        )
    snapshot_sha256 = _require_v2_sha256(
        current_authority.authority_snapshot_sha256,
        "current authority snapshot SHA-256",
    )
    owner_by_manifest: dict[str, str] = {}
    capture_game_by_id = {
        game.game_id: game for game in capture.games
    }
    for game_id in universe_game_ids:
        for receipt in capture_game_by_id[game_id].manifests:
            if receipt.manifest_sha256 in owner_by_manifest:
                raise X13MarketRecleanError(
                    "capture manifest belongs to multiple games"
                )
            owner_by_manifest[receipt.manifest_sha256] = game_id
    entries: list[VerifiedX13MarketManifestV2] = []
    for receipt in capture.manifests:
        entries.append(
            _read_x13_market_manifest_index_entry_v2(
                receipt,
                game_id=owner_by_manifest.get(receipt.manifest_sha256),
                current_authority=current_authority,
                raw_store_root=raw_store_root,
            )
        )
    manifest_ids = tuple(
        entry.receipt.manifest_sha256 for entry in entries
    )
    if (
        tuple(sorted(manifest_ids))
        != capture.pointer.raw_manifest_sha256s
        or len(set(manifest_ids)) != len(manifest_ids)
        or set(owner_by_manifest) - set(manifest_ids)
    ):
        raise X13MarketRecleanError(
            "payload-free index differs from capture pointer"
        )
    by_game: dict[str, tuple[VerifiedX13MarketManifestV2, ...]] = {}
    for game_id in universe_game_ids:
        game_entries = tuple(
            entry for entry in entries if entry.game_id == game_id
        )
        expected = {
            receipt.manifest_sha256
            for receipt in capture_game_by_id[game_id].manifests
        }
        if {
            entry.receipt.manifest_sha256 for entry in game_entries
        } != expected:
            raise X13MarketRecleanError(
                f"{game_id} payload-free manifest ownership is incomplete"
            )
        by_game[game_id] = game_entries
    batch_entries = tuple(
        entry for entry in entries if entry.game_id is None
    )
    if (
        len(batch_entries) != 1
        or batch_entries[0].resource != "kalshi_cutoff"
        or batch_entries[0].receipt.manifest_sha256
        != capture.kalshi_cutoff.manifest_sha256
    ):
        raise X13MarketRecleanError(
            "Kalshi historical cutoff manifest ownership is invalid"
        )
    return VerifiedX13MarketCaptureIndexV2(
        status="PASS",
        capture_sha256=_require_v2_sha256(
            capture.pointer.capture_sha256,
            "capture SHA-256",
        ),
        capture_receipt_sha256=_require_v2_sha256(
            capture.receipt.receipt_sha256,
            "capture receipt SHA-256",
        ),
        raw_manifest_count=len(entries),
        raw_byte_length=sum(
            entry.receipt.byte_length for entry in entries
        ),
        manifest_hashes_verified=True,
        current_authority_snapshot_sha256=snapshot_sha256,
        current_authority_component_sha256s=components,
        universe=universe,
        capture=capture,
        manifests_by_game=MappingProxyType(by_game),
        batch_manifests=batch_entries,
        raw_store_root=raw_store_root,
        program_root=program_root,
    )


def verify_x13_market_capture_v2(
    *,
    universe_artifact: str | Path,
    capture_pointer: str | Path,
    raw_store_root: str | Path,
    program_root: str | Path,
) -> VerifiedX13MarketCaptureIndexV2:
    """Verify the complete capture once and return no parsed raw payloads."""

    universe = load_x13_universe_artifact(universe_artifact)
    raw_root = Path(raw_store_root).resolve()
    program = Path(program_root).resolve()
    try:
        capture = load_x13_capture(
            capture_pointer,
            plans=universe.capture_plans,
            universe_id=universe.universe_id,
            raw_store_root=raw_root,
            program_root=program,
        )
        return _build_verified_x13_market_capture_index_v2(
            capture,
            universe=universe,
            raw_store_root=raw_root,
            program_root=program,
        )
    except X13MarketRecleanError:
        raise
    except (OSError, ValueError) as error:
        raise X13MarketRecleanError(str(error)) from error


def _reopen_x13_market_game_documents_v2(
    index: VerifiedX13MarketCaptureIndexV2,
    game_id: str,
) -> tuple[VerifiedCaptureDocumentV1, ...]:
    """Reopen, parse, and prove only one selected game's raw objects."""

    from prediction_market.sports.nfl_x13_pipeline import (
        VerifiedCaptureDocumentV1,
        _resource,
        _strict_json,
        _validate_game_capture_proofs,
    )
    from prediction_market.static_store import (
        StaticStoreError,
        read_verified_static_object,
    )

    entries = index.manifests_by_game.get(game_id)
    universe_by_game = {
        game.game_id: game for game in index.universe.games
    }
    capture_by_game = {
        game.game_id: game for game in index.capture.games
    }
    game = universe_by_game.get(game_id)
    capture_game = capture_by_game.get(game_id)
    if entries is None or game is None or capture_game is None:
        raise X13MarketRecleanError(
            "selected game lacks verified capture evidence"
        )
    documents: list[VerifiedCaptureDocumentV1] = []
    for entry in entries:
        try:
            verified = read_verified_static_object(
                entry.receipt.manifest_path,
                store_root=index.raw_store_root,
                program_root=index.program_root,
            )
        except (OSError, StaticStoreError) as error:
            raise X13MarketRecleanError(
                "selected game raw object verification failed"
            ) from error
        record = verified.record
        manifest = record.manifest
        if (
            entry.game_id != game_id
            or manifest.manifest_sha256
            != entry.receipt.manifest_sha256
            or manifest.object_sha256 != entry.receipt.object_sha256
            or manifest.byte_length != entry.receipt.byte_length
            or record.manifest_path != entry.receipt.manifest_path
            or record.object_path != entry.receipt.object_path
            or record.dataset != entry.dataset_id
            or record.version != entry.source_version
            or manifest.license_ref != entry.license_ref
            or manifest.license_status
            != entry.captured_license_status
            or _resource(manifest.source_url) != entry.resource
        ):
            raise X13MarketRecleanError(
                "selected game raw object differs from payload-free index"
            )
        payload = _strict_json(
            verified.object_bytes,
            context=f"capture {entry.receipt.manifest_sha256}",
            exact_decimal_numbers=entry.resource == "polymarket_trades",
        )
        documents.append(
            VerifiedCaptureDocumentV1(
                game_id=game_id,
                resource=entry.resource,
                manifest_sha256=manifest.manifest_sha256,
                object_sha256=manifest.object_sha256,
                byte_length=manifest.byte_length,
                source_url=manifest.source_url,
                source_request=MappingProxyType(
                    dict(manifest.source_request)
                ),
                source_cursor=manifest.source_cursor,
                coverage=manifest.coverage,
                dataset_id=manifest.dataset_id,
                license_ref=manifest.license_ref,
                license_status=manifest.license_status,
                current_license_status=entry.current_license_status,
                current_authority_snapshot_sha256=(
                    index.current_authority_snapshot_sha256
                ),
                current_catalog_registry_sha256=(
                    index.current_authority_component_sha256s[
                        "charter/catalog_registry.csv"
                    ]
                ),
                current_data_license_registry_sha256=(
                    index.current_authority_component_sha256s[
                        "registries/data_license_register.csv"
                    ]
                ),
                current_dataset_registry_sha256=(
                    index.current_authority_component_sha256s[
                        "registries/dataset_registry.csv"
                    ]
                ),
                current_experiment_registry_sha256=(
                    index.current_authority_component_sha256s[
                        "registries/experiment_registry.csv"
                    ]
                ),
                payload=payload,
            )
        )
    values = tuple(documents)
    _validate_game_capture_proofs(capture_game, game, values)
    return values


def reclean_x13_verified_market_game_v2(
    index: VerifiedX13MarketCaptureIndexV2,
    game_id: str,
) -> X13MarketRecleanGameV2:
    """Reopen and reclean one selected game from a verified frozen index."""

    if not isinstance(index, VerifiedX13MarketCaptureIndexV2):
        raise TypeError("index must be VerifiedX13MarketCaptureIndexV2")
    if index.status != "PASS" or index.manifest_hashes_verified is not True:
        raise X13MarketRecleanError("capture index is not verified")
    if type(game_id) is not str or not game_id:
        raise TypeError("game_id must be nonempty text")
    universe_by_game = {
        game.game_id: game for game in index.universe.games
    }
    capture_by_game = {
        game.game_id: game for game in index.capture.games
    }
    game = universe_by_game.get(game_id)
    capture_game = capture_by_game.get(game_id)
    if (
        game is None
        or capture_game is None
        or game_id not in index.manifests_by_game
    ):
        raise X13MarketRecleanError(
            "selected game is absent from the verified capture index"
        )
    documents = _reopen_x13_market_game_documents_v2(index, game_id)
    return _reclean_x13_market_game_documents_v2(
        game=game,
        capture_game=capture_game,
        documents=documents,
        capture_sha256=index.capture_sha256,
    )


__all__ = [
    "MarketCleaningAuditRowV1",
    "MarketCleaningAuditV1",
    "MarketCleaningError",
    "MarketCleaningInputV1",
    "build_polymarket_cleaning_input_v1",
    "clean_market_rows_v1",
    "deduplicate_cleaned_observations_v1",
    "VerifiedX13MarketCaptureIndexV2",
    "VerifiedX13MarketManifestV2",
    "X13MarketCleaningAuditRowV2",
    "X13MarketRecleanError",
    "X13MarketRecleanGameV2",
    "reclean_x13_verified_market_game_v2",
    "verify_x13_market_capture_v2",
]
