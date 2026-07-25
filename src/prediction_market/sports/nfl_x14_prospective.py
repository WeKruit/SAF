"""Governed prospective NFL game-feed and market-L2 capture for X-14."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence

from prediction_market.raw_store import (
    PartitionBoundaryError,
    RawSegmentWriter,
    SegmentManifest,
)


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_MARKET_KINDS = frozenset(
    {"market_snapshot", "market_delta", "bbo", "trade"}
)
_SPORTS_KINDS = frozenset({"play_revision"})
_EPOCH_REASONS = frozenset({"reconnect", "sequence_gap"})


class CaptureMode(str, Enum):
    """Input lineage classes that may never share one association run."""

    HISTORICAL_TRADES_ONLY = "historical_trades_only"
    PROSPECTIVE_L2 = "prospective_l2"


class CaptureContinuityError(ValueError):
    """The source stream cannot be continued without a new epoch."""


def _identifier(value: str, field: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a stable non-empty identifier")
    return value


def _utc_text(value: str, field: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError(f"{field} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field} must be a canonical UTC timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must be a canonical UTC timestamp")
    if parsed.isoformat().replace("+00:00", "Z") != value:
        raise ValueError(f"{field} must be a canonical UTC timestamp")
    return value


def _unique_sorted(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be a sequence of stable identifiers")
    normalized = tuple(_identifier(value, field) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} cannot contain duplicates")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class ProspectiveSessionSpec:
    """Frozen single-game universe and capture boundary contract."""

    experiment_id: str
    session_id: str
    game_id: str
    scheduled_start_utc: str
    away_team: str
    home_team: str
    identity_key: str
    polymarket_event_id: str
    polymarket_event_slug: str
    polymarket_native_game_id: str
    kalshi_event_id: str
    venue_date_suffix_is_identity: bool
    polymarket_condition_ids: tuple[str, ...]
    polymarket_token_ids: tuple[str, ...]
    kalshi_market_ids: tuple[str, ...]
    kalshi_enabled: bool
    start_boundary: Literal["market_creation"]
    end_boundary: Literal["market_resolution"]

    @classmethod
    def freeze(
        cls,
        *,
        session_id: str,
        game_id: str,
        scheduled_start_utc: str,
        away_team: str,
        home_team: str,
        polymarket_event_id: str,
        polymarket_event_slug: str,
        polymarket_native_game_id: str,
        kalshi_event_id: str,
        polymarket_condition_ids: Sequence[str],
        discovered_single_game_condition_ids: Sequence[str],
        polymarket_token_ids: Sequence[str],
        discovered_single_game_token_ids: Sequence[str],
        kalshi_credentials_available: bool,
        kalshi_market_ids: Sequence[str],
    ) -> ProspectiveSessionSpec:
        """Freeze the complete token universe before any association analysis."""

        if type(kalshi_credentials_available) is not bool:
            raise ValueError("kalshi_credentials_available must be boolean")
        scheduled = _utc_text(
            scheduled_start_utc, "scheduled_start_utc"
        )
        away = _identifier(away_team, "away_team")
        home = _identifier(home_team, "home_team")
        if away == home:
            raise ValueError("away_team and home_team must differ")
        selected_conditions = _unique_sorted(
            polymarket_condition_ids, "polymarket_condition_ids"
        )
        discovered_conditions = _unique_sorted(
            discovered_single_game_condition_ids,
            "discovered_single_game_condition_ids",
        )
        if (
            not selected_conditions
            or selected_conditions != discovered_conditions
        ):
            raise ValueError(
                "polymarket condition universe must be non-empty and "
                "exactly equal to the discovered single-game condition "
                "universe"
            )
        selected = _unique_sorted(
            polymarket_token_ids, "polymarket_token_ids"
        )
        discovered = _unique_sorted(
            discovered_single_game_token_ids,
            "discovered_single_game_token_ids",
        )
        if not selected:
            raise ValueError(
                "polymarket_token_ids must include every single-game token"
            )
        if selected != discovered:
            raise ValueError(
                "polymarket token universe must be exactly equal to the "
                "discovered single-game token universe"
            )
        kalshi = _unique_sorted(kalshi_market_ids, "kalshi_market_ids")
        if kalshi and not kalshi_credentials_available:
            raise ValueError(
                "Kalshi capture cannot be enabled without credentials"
            )
        return cls(
            experiment_id="X-14",
            session_id=_identifier(session_id, "session_id"),
            game_id=_identifier(game_id, "game_id"),
            scheduled_start_utc=scheduled,
            away_team=away,
            home_team=home,
            identity_key=f"{scheduled}|{away}|{home}",
            polymarket_event_id=_identifier(
                polymarket_event_id, "polymarket_event_id"
            ),
            polymarket_event_slug=_identifier(
                polymarket_event_slug, "polymarket_event_slug"
            ),
            polymarket_native_game_id=_identifier(
                polymarket_native_game_id,
                "polymarket_native_game_id",
            ),
            kalshi_event_id=_identifier(
                kalshi_event_id, "kalshi_event_id"
            ),
            venue_date_suffix_is_identity=False,
            polymarket_condition_ids=selected_conditions,
            polymarket_token_ids=selected,
            kalshi_market_ids=kalshi,
            kalshi_enabled=bool(kalshi),
            start_boundary="market_creation",
            end_boundary="market_resolution",
        )


@dataclass(frozen=True)
class ProspectiveCaptureRecord:
    """One canonical envelope around an exact native source payload."""

    experiment_id: str
    session_id: str
    game_id: str
    epoch: int
    capture_mode: CaptureMode
    record_class: Literal["market", "sports"]
    event_kind: str
    source: str
    source_received_time: str
    local_received_time: str
    source_sequence: int | None
    market_id: str | None
    condition_id: str | None
    token_id: str | None
    play_id: str | None
    play_revision: int | None
    native_payload: bytes

    def __post_init__(self) -> None:
        if self.experiment_id != "X-14":
            raise ValueError("experiment_id must be X-14")
        _identifier(self.session_id, "session_id")
        _identifier(self.game_id, "game_id")
        _identifier(self.source, "source")
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if not isinstance(self.capture_mode, CaptureMode):
            raise ValueError("capture_mode must be a CaptureMode")
        _utc_text(self.source_received_time, "source_received_time")
        _utc_text(self.local_received_time, "local_received_time")
        if self.source_sequence is not None and (
            type(self.source_sequence) is not int
            or self.source_sequence < 0
        ):
            raise ValueError(
                "source_sequence must be a non-negative integer or null"
            )
        if type(self.native_payload) is not bytes:
            raise ValueError("native_payload must be exact bytes")

        if self.record_class == "market":
            if self.event_kind not in _MARKET_KINDS:
                raise ValueError("unsupported market event_kind")
            if self.source not in {"polymarket", "kalshi"}:
                raise ValueError(
                    "market source must be polymarket or kalshi"
                )
            if self.market_id is None:
                raise ValueError("market record requires market_id")
            _identifier(self.market_id, "market_id")
            if self.source == "polymarket":
                if self.condition_id is None or self.token_id is None:
                    raise ValueError(
                        "Polymarket record requires condition_id and token_id"
                    )
                _identifier(self.condition_id, "condition_id")
                _identifier(self.token_id, "token_id")
            elif (
                self.condition_id is not None or self.token_id is not None
            ):
                raise ValueError(
                    "Kalshi record cannot contain Polymarket identifiers"
                )
            if self.play_id is not None or self.play_revision is not None:
                raise ValueError(
                    "market record cannot contain sports play fields"
                )
        elif self.record_class == "sports":
            if self.event_kind not in _SPORTS_KINDS:
                raise ValueError("unsupported sports event_kind")
            if self.play_id is None:
                raise ValueError("sports record requires play_id")
            _identifier(self.play_id, "play_id")
            if type(self.play_revision) is not int or self.play_revision < 0:
                raise ValueError(
                    "sports record requires non-negative play_revision"
                )
            if (
                self.market_id is not None
                or self.condition_id is not None
                or self.token_id is not None
                or self.source_sequence is not None
            ):
                raise ValueError(
                    "sports record cannot contain market identity or sequence"
                )
        else:
            raise ValueError("record_class must be market or sports")

    def to_bytes(self) -> bytes:
        payload_sha256 = "sha256:" + hashlib.sha256(
            self.native_payload
        ).hexdigest()
        document = {
            "capture_contract": "nfl-x14-prospective/v0",
            "capture_mode": self.capture_mode.value,
            "condition_id": self.condition_id,
            "epoch": self.epoch,
            "event_kind": self.event_kind,
            "experiment_id": self.experiment_id,
            "game_id": self.game_id,
            "local_received_time": self.local_received_time,
            "market_id": self.market_id,
            "native_payload_base64": base64.b64encode(
                self.native_payload
            ).decode("ascii"),
            "native_payload_sha256": payload_sha256,
            "play_id": self.play_id,
            "play_revision": self.play_revision,
            "record_class": self.record_class,
            "session_id": self.session_id,
            "source": self.source,
            "source_received_time": self.source_received_time,
            "source_sequence": self.source_sequence,
            "token_id": self.token_id,
        }
        return json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


class ProspectiveCaptureValidator:
    """Enforce snapshot/delta continuity within explicit connection epochs."""

    def __init__(self, *, session_id: str, game_id: str) -> None:
        self.session_id = _identifier(session_id, "session_id")
        self.game_id = _identifier(game_id, "game_id")
        self.epoch = 0
        self._last_sequence: int | None = None
        self._snapshot_seen = False
        self._blocked = False
        self._play_revisions: dict[str, int] = {}

    def observe(self, record: ProspectiveCaptureRecord) -> None:
        if record.session_id != self.session_id:
            raise CaptureContinuityError("record session mismatch")
        if record.game_id != self.game_id:
            raise CaptureContinuityError("record game mismatch")
        if record.epoch != self.epoch:
            raise CaptureContinuityError(
                f"record epoch {record.epoch} does not match active epoch "
                f"{self.epoch}"
            )
        if self._blocked:
            raise CaptureContinuityError(
                "epoch is blocked; a gap cannot be bridged"
            )
        if record.record_class == "sports":
            assert record.play_id is not None
            assert record.play_revision is not None
            prior_revision = self._play_revisions.get(record.play_id)
            if (
                prior_revision is not None
                and record.play_revision < prior_revision
            ):
                self._blocked = True
                raise CaptureContinuityError(
                    f"play revision regressed for {record.play_id}: "
                    f"{prior_revision} to {record.play_revision}"
                )
            self._play_revisions[record.play_id] = record.play_revision
            return
        if record.capture_mode is not CaptureMode.PROSPECTIVE_L2:
            self._blocked = True
            raise CaptureContinuityError(
                "prospective capture cannot contain historical records"
            )
        if (
            record.event_kind == "market_delta"
            and not self._snapshot_seen
        ):
            self._blocked = True
            raise CaptureContinuityError(
                "market_delta requires a snapshot in the same epoch"
            )
        if record.source_sequence is not None:
            expected = (
                None
                if self._last_sequence is None
                else self._last_sequence + 1
            )
            if expected is not None and record.source_sequence != expected:
                self._blocked = True
                raise CaptureContinuityError(
                    f"sequence gap in epoch {self.epoch}: "
                    f"expected {expected}, observed {record.source_sequence}"
                )
            self._last_sequence = record.source_sequence
        if record.event_kind == "market_snapshot":
            self._snapshot_seen = True

    def start_new_epoch(self, *, reason: str) -> int:
        if reason not in _EPOCH_REASONS:
            raise ValueError(
                "new epoch reason must be reconnect or sequence_gap"
            )
        self.epoch += 1
        self._last_sequence = None
        self._snapshot_seen = False
        self._blocked = False
        self._play_revisions = {}
        return self.epoch


def validate_association_provenance(
    records: Sequence[ProspectiveCaptureRecord],
) -> CaptureMode:
    """Reject an association input that crosses historical/prospective lineage."""

    if not records:
        raise ValueError("association input cannot be empty")
    modes = {record.capture_mode for record in records}
    if len(modes) != 1:
        raise ValueError(
            "historical trades-only and prospective L2 cannot be mixed"
        )
    return next(iter(modes))


class ProspectiveCaptureSession:
    """Persist validated X-14 envelopes into immutable hourly raw segments."""

    def __init__(
        self,
        *,
        raw_root: str | Path,
        source: str,
        stream: str,
        spec: ProspectiveSessionSpec,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.source = _identifier(source, "source")
        self.stream = _identifier(stream, "stream")
        self.spec = spec
        self.validator = ProspectiveCaptureValidator(
            session_id=spec.session_id,
            game_id=spec.game_id,
        )
        self._writer: RawSegmentWriter | None = None
        self._manifests: list[SegmentManifest] = []
        self._closed = False

    def _new_writer(self) -> RawSegmentWriter:
        return RawSegmentWriter(
            self.raw_root,
            source=self.source,
            stream=self.stream,
            capture_session_id=self.spec.session_id,
        )

    def _append_bytes(self, payload: bytes, receive_at: str) -> None:
        if self._writer is None:
            self._writer = self._new_writer()
        try:
            self._writer.append(payload, receive_at=receive_at)
        except PartitionBoundaryError:
            self._manifests.append(self._writer.seal())
            self._writer = self._new_writer()
            self._writer.append(payload, receive_at=receive_at)

    def append(self, record: ProspectiveCaptureRecord) -> None:
        if self._closed:
            raise CaptureContinuityError("capture session is closed")
        if record.source != self.source:
            raise CaptureContinuityError("record source mismatch")
        self.validator.observe(record)
        payload = record.to_bytes()
        self._append_bytes(payload, record.local_received_time)

    def start_new_epoch(self, *, reason: str) -> int:
        if self._closed:
            raise CaptureContinuityError("capture session is closed")
        if self._writer is not None:
            self._manifests.append(self._writer.seal())
            self._writer = None
        return self.validator.start_new_epoch(reason=reason)

    def close(self) -> tuple[SegmentManifest, ...]:
        if self._closed:
            raise CaptureContinuityError("capture session is closed")
        if self._writer is not None:
            self._manifests.append(self._writer.seal())
            self._writer = None
        self._closed = True
        return tuple(self._manifests)


__all__ = [
    "CaptureContinuityError",
    "CaptureMode",
    "ProspectiveCaptureRecord",
    "ProspectiveCaptureSession",
    "ProspectiveCaptureValidator",
    "ProspectiveSessionSpec",
    "validate_association_provenance",
]
