"""Sports-expert semantic review records for the frozen X-13 factor registry.

These records are deliberately a closure gate, not a signal shortlist.  They
capture the sports meaning that must be resolved before any later shortlist can
be proposed from historical source-time evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from prediction_market.sports.nfl_factor_lab_analysis import (
    HumanFactorReviewV1,
    ShortlistLockV1,
    freeze_factor_shortlist as _freeze_factor_shortlist,
)


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FACTOR_ID_RE = re.compile(r"NFL(?:\.[A-Z0-9_]+)+\Z")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_UTC_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_CANDIDATE_REVIEW_SCHEMA = "nfl_x13_candidate_expert_review_v1"
_REVIEW_DECISIONS = frozenset(
    {"ACCEPT", "REDEFINE", "DATA_GAP", "REJECT"}
)
_REVIEW_HORIZONS = frozenset({1, 2, 5, 10, 30, 60})

FACTOR_REGISTRY_V2_SHA256 = (
    "sha256:295e4e48237f8523fe3153f28b5b28adc9f6e1d3a219568047da5c250ed243c1"
)
CANONICAL_FACTOR_REGISTRY_V2_PATH = (
    Path(__file__).resolve().parents[3]
    / "registries/factors/nfl_factor_registry_v2.json"
)
X13_EXPERT_REVIEW_ALLOWED_CONTROL_FIELDS = frozenset(
    {
        "drive_id",
        "field_goal_result",
        "fumble_recovery_team",
        "game_seconds_remaining",
        "possession_team",
        "pre_distance",
        "pre_down",
        "pre_event_probability",
        "pre_red_zone",
        "pre_score_differential",
        "punt_result",
        "return_yards",
        "terminal_drive_result",
        "turnover",
    }
)


class SportsExpertReviewError(ValueError):
    """A proposed sports-expert review is not auditable."""


class ReviewDecision(str, Enum):
    """The only semantic-review outcomes permitted before a shortlist."""

    ACCEPT = "ACCEPT"
    REDEFINE = "REDEFINE"
    DATA_GAP = "DATA_GAP"
    REJECT = "REJECT"


ReviewStatus = ReviewDecision


@dataclass(frozen=True, slots=True)
class CandidateExpertReviewArtifactV1:
    """Content-addressed reference to one independently supplied review."""

    artifact_path: Path
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedCandidateExpertReviewV1:
    """Verified candidate evidence that can be converted to the actual gate type."""

    artifact_path: Path
    artifact_sha256: str
    experiment_id: str
    factor_id: str
    factor_version: str
    market_family: str
    horizon_seconds: int
    decision: str
    reviewer: str
    reviewed_at_utc: str
    sports_rationale: str
    anomaly_case_ids: tuple[str, ...]
    priority: str
    discovery_sha256: str
    factor_registry_sha256: str
    matched_sha256: str

    def to_human_factor_review(self) -> HumanFactorReviewV1:
        """Adapt only verified artifact fields to the existing shortlist gate."""

        return HumanFactorReviewV1(
            factor_id=self.factor_id,
            factor_version=self.factor_version,
            market_family=self.market_family,
            horizon_seconds=self.horizon_seconds,
            decision=self.decision,
            sports_rationale=self.sports_rationale,
            anomaly_case_ids=self.anomaly_case_ids,
            priority=self.priority,
            reviewer=self.reviewer,
            reviewed_at_utc=self.reviewed_at_utc,
        )


def _nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SportsExpertReviewError(f"{field} must be non-empty text")
    return value.strip()


def _factor_id(value: object, *, field: str) -> str:
    text = _nonempty_text(value, field=field)
    if _FACTOR_ID_RE.fullmatch(text) is None:
        raise SportsExpertReviewError(f"{field} must be an NFL factor id")
    return text


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SportsExpertReviewError(f"{field} must be a sha256 digest")
    return value


def _unique_texts(
    values: object,
    *,
    field: str,
    parser: Any = _nonempty_text,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise SportsExpertReviewError(f"{field} must be a tuple")
    parsed = tuple(parser(value, field=field) for value in values)
    if not parsed:
        raise SportsExpertReviewError(f"{field} must not be empty")
    if len(set(parsed)) != len(parsed):
        raise SportsExpertReviewError(f"{field} must not contain duplicates")
    return tuple(sorted(parsed))


def load_x13_frozen_factor_registry(path: Path) -> dict[str, str]:
    """Load the only factor-id/hash bindings accepted by the X-13 review."""

    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise SportsExpertReviewError("frozen factor registry must be a file")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SportsExpertReviewError(
            "frozen factor registry is not valid JSON"
        ) from error
    if "sha256:" + hashlib.sha256(raw).hexdigest() != FACTOR_REGISTRY_V2_SHA256:
        raise SportsExpertReviewError(
            "frozen factor registry physical hash does not match canonical X-13"
        )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "nfl_factor_registry_v2"
        or payload.get("experiment_id") != "X-13"
    ):
        raise SportsExpertReviewError("frozen factor registry has wrong schema")
    definitions = payload.get("factor_definitions")
    if not isinstance(definitions, list) or not definitions:
        raise SportsExpertReviewError("frozen factor registry has no definitions")
    registry: dict[str, str] = {}
    for item in definitions:
        if not isinstance(item, Mapping):
            raise SportsExpertReviewError("frozen factor definition is malformed")
        factor_id = _factor_id(item.get("factor_id"), field="factor_id")
        definition_sha256 = _digest(
            item.get("definition_sha256"),
            field="definition_sha256",
        )
        if factor_id in registry:
            raise SportsExpertReviewError(
                "frozen factor registry has duplicate factor_id"
            )
        registry[factor_id] = definition_sha256
    return registry


def _validate_record_bindings(
    record: "ExpertReviewRecord",
    *,
    frozen_factor_registry: Mapping[str, str],
    allowed_control_fields: frozenset[str],
) -> None:
    expected_digest = frozen_factor_registry.get(record.factor_id)
    if expected_digest is None:
        raise SportsExpertReviewError(
            "factor_id is absent from frozen_factor_registry"
        )
    if record.definition_sha256 != expected_digest:
        raise SportsExpertReviewError(
            "definition_sha256 does not match frozen_factor_registry"
        )
    unknown_overlap = sorted(
        set(record.overlap_factor_ids) - set(frozen_factor_registry)
    )
    if unknown_overlap:
        raise SportsExpertReviewError(
            "overlap_factor_ids are absent from frozen_factor_registry"
        )
    unsupported_controls = sorted(
        set(record.required_control_fields) - allowed_control_fields
    )
    if unsupported_controls:
        raise SportsExpertReviewError(
            "required_control_fields are outside the explicit allowed set"
        )


@dataclass(frozen=True, slots=True)
class ExpertReviewRecord:
    """A factor-level football semantic decision bound to its definition hash."""

    factor_id: str
    definition_sha256: str
    decision: ReviewDecision
    actor: str
    beneficiary: str
    focal_orientation: str
    overlap_factor_ids: tuple[str, ...]
    required_control_fields: tuple[str, ...]
    rationale_zh: str
    shortlist_eligible: bool = False

    def __post_init__(self) -> None:
        factor_id = _factor_id(self.factor_id, field="factor_id")
        object.__setattr__(self, "factor_id", factor_id)
        object.__setattr__(
            self,
            "definition_sha256",
            _digest(self.definition_sha256, field="definition_sha256"),
        )
        if not isinstance(self.decision, ReviewDecision):
            raise SportsExpertReviewError("decision must be a ReviewDecision")
        object.__setattr__(self, "actor", _nonempty_text(self.actor, field="actor"))
        object.__setattr__(
            self,
            "beneficiary",
            _nonempty_text(self.beneficiary, field="beneficiary"),
        )
        object.__setattr__(
            self,
            "focal_orientation",
            _nonempty_text(self.focal_orientation, field="focal_orientation"),
        )
        overlap = _unique_texts(
            self.overlap_factor_ids,
            field="overlap_factor_ids",
            parser=_factor_id,
        )
        if factor_id in overlap:
            raise SportsExpertReviewError(
                "overlap_factor_ids must not contain factor_id"
            )
        object.__setattr__(self, "overlap_factor_ids", overlap)
        object.__setattr__(
            self,
            "required_control_fields",
            _unique_texts(
                self.required_control_fields,
                field="required_control_fields",
            ),
        )
        rationale = _nonempty_text(self.rationale_zh, field="rationale_zh")
        if _CJK_RE.search(rationale) is None:
            raise SportsExpertReviewError("rationale_zh must contain Chinese text")
        object.__setattr__(self, "rationale_zh", rationale)
        if self.shortlist_eligible is not False:
            raise SportsExpertReviewError(
                "shortlist_eligible must remain false until a separate gate"
            )

    def as_payload(self) -> dict[str, object]:
        return {
            "factor_id": self.factor_id,
            "definition_sha256": self.definition_sha256,
            "decision": self.decision.value,
            "actor": self.actor,
            "beneficiary": self.beneficiary,
            "focal_orientation": self.focal_orientation,
            "overlap_factor_ids": list(self.overlap_factor_ids),
            "required_control_fields": list(self.required_control_fields),
            "rationale_zh": self.rationale_zh,
            "shortlist_eligible": False,
        }


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    """Hash canonical payload material, excluding its self-referential digest."""

    material = dict(payload)
    material.pop("payload_sha256", None)
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_x13_sports_expert_review_payload(
    records: Iterable[ExpertReviewRecord],
) -> dict[str, object]:
    """Publish a review only against the byte-verified canonical registry."""

    frozen = load_x13_frozen_factor_registry(
        CANONICAL_FACTOR_REGISTRY_V2_PATH
    )
    verified = tuple(records)
    if not verified:
        raise SportsExpertReviewError("at least one review record is required")
    if not all(isinstance(record, ExpertReviewRecord) for record in verified):
        raise SportsExpertReviewError("records must contain ExpertReviewRecord values")
    factor_ids = [record.factor_id for record in verified]
    if len(set(factor_ids)) != len(factor_ids):
        raise SportsExpertReviewError("review factor ids must be unique")
    for record in verified:
        _validate_record_bindings(
            record,
            frozen_factor_registry=frozen,
            allowed_control_fields=X13_EXPERT_REVIEW_ALLOWED_CONTROL_FIELDS,
        )
    reviews = [record.as_payload() for record in sorted(verified, key=lambda item: item.factor_id)]
    payload: dict[str, object] = {
        "schema_version": "NFLX13SportsExpertReviewV2",
        "experiment_id": "X-13",
        "claim_boundary": "PRELIMINARY_SOURCE_TIME_ONLY",
        "source_factor_registry": "nfl_factor_registry_v2.json",
        "source_factor_registry_sha256": FACTOR_REGISTRY_V2_SHA256,
        "supersedes": ["nfl_factor_registry_v2.json"],
        "shortlist_status": "CLOSED_BY_DEFAULT",
        "reviews": reviews,
    }
    payload["payload_sha256"] = canonical_payload_sha256(payload)
    return payload


def _review(
    factor_id: str,
    definition_sha256: str,
    decision: ReviewDecision,
    actor: str,
    beneficiary: str,
    focal_orientation: str,
    overlap_factor_ids: tuple[str, ...],
    rationale_zh: str,
    controls: tuple[str, ...] = (
        "pre_event_probability",
        "game_seconds_remaining",
        "pre_score_differential",
        "possession_team",
    ),
) -> ExpertReviewRecord:
    return ExpertReviewRecord(
        factor_id=factor_id,
        definition_sha256=definition_sha256,
        decision=decision,
        actor=actor,
        beneficiary=beneficiary,
        focal_orientation=focal_orientation,
        overlap_factor_ids=overlap_factor_ids,
        required_control_fields=controls,
        rationale_zh=rationale_zh,
    )


def seed_x13_sports_expert_reviews() -> tuple[ExpertReviewRecord, ...]:
    """Return the current, deliberately conservative football review calls."""

    return (
        _review(
            "NFL.EVENT.SACK",
            "sha256:97c2ca78764f2ebc42152ec12350e757bfe462f23e66bcf211bc82f78717baba",
            ReviewDecision.ACCEPT,
            "防守方",
            "防守方",
            "以受益防守球队为焦点；被擒杀进攻方为反向。",
            ("NFL.EVENT.PASS",),
            "擒杀是最终可核验的原子比赛事件。",
        ),
        _review(
            "NFL.TURNOVER.INTERCEPTION",
            "sha256:14dcf7682a33d9f258767a0bda4ae68f8501b0627fb1ce8c7508ee6a5506489e",
            ReviewDecision.ACCEPT,
            "传球进攻方",
            "截球防守方",
            "以获球防守球队为焦点；原进攻方为反向。",
            ("NFL.TURNOVER.FUMBLE_LOST",),
            "截球有明确失球与获球双方，是原子事件。",
        ),
        _review(
            "NFL.TURNOVER.FUMBLE_LOST",
            "sha256:af7632709db0bf15f0258de9a2786f9319e56ee9267729db8cc5b9fb5f1ddad8",
            ReviewDecision.ACCEPT,
            "掉球失去球权方",
            "获得球权方",
            "以获球球队为焦点；失球方为反向。",
            ("NFL.EVENT.FUMBLE_RECOVERY",),
            "失去球权已完成，双方方向可确定。",
        ),
        _review(
            "NFL.TURNOVER.ON_DOWNS",
            "sha256:86b97eb5718c0b28824cd1bfcd226ed8b5286b47d398ad557b1a8a70da09f836",
            ReviewDecision.ACCEPT,
            "未转换进攻方",
            "取得球权防守方",
            "以取得球权球队为焦点；未转换方为反向。",
            ("NFL.EVENT.FOURTH_DOWN_CONVERSION",),
            "四档失误导致球权转换，事件边界明确。",
        ),
        _review(
            "NFL.SCORE.PASS_TD",
            "sha256:8cb388633a43e558fbe1bfebd962558b16470f27838255292242f1037cabec5c",
            ReviewDecision.ACCEPT,
            "达阵进攻方",
            "达阵进攻方",
            "以得分球队为焦点；对手为反向。",
            ("NFL.SCORE.RUSH_TD", "NFL.SCORE.RETURN_TD"),
            "传球达阵是最终得分原子事件。",
        ),
        _review(
            "NFL.SCORE.RUSH_TD",
            "sha256:421e7fab43878cdad839bb54dd36fe5006cd114ede15652eeacc2f273ab2be88",
            ReviewDecision.ACCEPT,
            "达阵进攻方",
            "达阵进攻方",
            "以得分球队为焦点；对手为反向。",
            ("NFL.SCORE.PASS_TD", "NFL.SCORE.RETURN_TD"),
            "冲球达阵是最终得分原子事件。",
        ),
        _review(
            "NFL.SCORE.RETURN_TD",
            "sha256:fd15e3f25fd9cf041f4d43d10e101dfe9c8701fb8115c8f00445795e50250d2c",
            ReviewDecision.ACCEPT,
            "回攻得分方",
            "回攻得分方",
            "以得分球队为焦点；对手为反向。",
            ("NFL.SCORE.PASS_TD", "NFL.SCORE.RUSH_TD"),
            "回攻达阵是最终得分原子事件。",
        ),
        _review(
            "NFL.SPECIAL.SAFETY",
            "sha256:fe83e77d5717d2e449bb0cce3409cfe3e6caea11a61c269055f185a3338ff8d4",
            ReviewDecision.ACCEPT,
            "被判安全分进攻方",
            "防守得分方",
            "以获得安全分球队为焦点；被判方为反向。",
            ("NFL.SCORE.RETURN_TD",),
            "安全分有最终计分和清晰受益方。",
        ),
        _review(
            "NFL.SEQUENCE.RED_ZONE_EMPTY_DRIVE",
            "sha256:4a199a3fdf7431ddd5495e0b20b70e53aa7019eda4c5e542e83dc899ec33f5a8",
            ReviewDecision.REDEFINE,
            "红区进攻方",
            "对方防守方",
            "以未得分红区进攻方为反向；须以驱动最终状态定向。",
            ("NFL.COMBO.RED_ZONE_TURNOVER", "NFL.STATE.FIELD_RED_ZONE"),
            "红区空回合必须按完整 drive 结算，不能拿单个 play 代替。",
            (
                "drive_id",
                "game_seconds_remaining",
                "possession_team",
                "pre_event_probability",
                "pre_red_zone",
                "terminal_drive_result",
            ),
        ),
        _review(
            "NFL.EVENT.FUMBLE_RECOVERY",
            "sha256:348d8b89338337c5a6ebfa140996b45d6ef51ba4e5ddf9d94929b4ed78997374",
            ReviewDecision.REDEFINE,
            "首次恢复球队",
            "首次恢复球队",
            "以恢复球队为焦点；须区分本方恢复与对手恢复。",
            ("NFL.TURNOVER.FUMBLE_LOST",),
            "恢复不必然换权，需先区分本方恢复和对手恢复。",
            (
                "fumble_recovery_team",
                "game_seconds_remaining",
                "possession_team",
                "pre_event_probability",
                "turnover",
            ),
        ),
        _review(
            "NFL.COMBO.THIRD_FOURTH_DOWN_FAILURE",
            "sha256:3f01d6aaaccd0e8ae60599cdd2d7fc60e31cead41b437658b626e697eef24954",
            ReviewDecision.REDEFINE,
            "未转换进攻方",
            "对方防守方",
            "以未转换进攻方为反向；三档与四档不得合并解释。",
            ("NFL.TURNOVER.ON_DOWNS", "NFL.EVENT.FOURTH_DOWN_CONVERSION"),
            "三档失败仍可继续，四档失败可能换权，必须拆分。",
            (
                "game_seconds_remaining",
                "possession_team",
                "pre_distance",
                "pre_down",
                "pre_event_probability",
                "turnover",
            ),
        ),
        _review(
            "NFL.SPECIAL.PUNT_POSITIVE_RETURN",
            "sha256:6f0a5ad732d6f8aece1690c3131ae69609af27848ccf12a191710e5b2abca98b",
            ReviewDecision.REDEFINE,
            "回攻方",
            "回攻方",
            "以回攻接球球队为焦点；踢球队为反向。",
            ("NFL.SPECIAL.PUNT", "NFL.SPECIAL.PUNT_TOUCHBACK"),
            "弃踢子类型的受益方不同，必须按落点和回攻结果拆分。",
            (
                "game_seconds_remaining",
                "possession_team",
                "pre_event_probability",
                "punt_result",
                "return_yards",
            ),
        ),
        _review(
            "NFL.SCORE.FG_MADE",
            "sha256:2e51ea63495554832191bf7cceacbb04b444e1dd7317e2c1dc2c82c2dc79d021",
            ReviewDecision.REDEFINE,
            "踢球得分方",
            "踢球得分方",
            "命中以得分球队为焦点；失误和封堵须以受益防守方为焦点。",
            ("NFL.SCORE.FG_BLOCKED", "NFL.SCORE.FG_MISSED"),
            "射门结果方向不对称，命中、失误和封堵不能共用方向。",
            (
                "field_goal_result",
                "game_seconds_remaining",
                "possession_team",
                "pre_event_probability",
            ),
        ),
        _review(
            "NFL.SCORE.FG_MISSED",
            "sha256:e0dd70d615d63501f2a9183b63f58ebc7b8bfff1d0b304b5e57b3faec00e28dc",
            ReviewDecision.REDEFINE,
            "踢球失误方",
            "对方防守方",
            "以受益防守球队为焦点；踢球失误方为反向。",
            ("NFL.SCORE.FG_BLOCKED", "NFL.SCORE.FG_MADE"),
            "射门失误的市场方向应由获益对手定义。",
            (
                "field_goal_result",
                "game_seconds_remaining",
                "possession_team",
                "pre_event_probability",
            ),
        ),
        _review(
            "NFL.SCORE.FG_BLOCKED",
            "sha256:222956e7156c4c5f1cfede84da161a3fc651f867662afa5670fed1e464d85275",
            ReviewDecision.REDEFINE,
            "封堵防守方",
            "封堵防守方",
            "以封堵球队为焦点；踢球队为反向。",
            ("NFL.SCORE.FG_MADE", "NFL.SCORE.FG_MISSED"),
            "封堵含防守行动，不能与普通未命中混写。",
            (
                "field_goal_result",
                "game_seconds_remaining",
                "possession_team",
                "pre_event_probability",
            ),
        ),
        _review(
            "NFL.SCORE.PAT_MADE",
            "sha256:a53e66177a8fc66c611c54c0216b71e455ac570789b74caacde2c609a86ebaad",
            ReviewDecision.REJECT,
            "踢球队",
            "踢球队",
            "不进入独立候选。",
            ("NFL.SEQUENCE.TD_TRY_FINALIZED",),
            "加分尝试依附达阵，不能作为独立语义事件。",
        ),
        _review(
            "NFL.SCORE.PAT_MISSED",
            "sha256:e542779926a5667772c8c23e814b28a6401a3702ef85c023060677cc9b28c501",
            ReviewDecision.REJECT,
            "踢球队",
            "对方球队",
            "不进入独立候选。",
            ("NFL.SEQUENCE.TD_TRY_FINALIZED",),
            "加分尝试依附达阵，不能作为独立语义事件。",
        ),
        _review(
            "NFL.SCORE.PAT_BLOCKED",
            "sha256:ea1f9622af8fb67f62229843462df1b8e3cbb4a07f4d6d2cf3ef7ed755d556d2",
            ReviewDecision.REJECT,
            "封堵方",
            "封堵方",
            "不进入独立候选。",
            ("NFL.SEQUENCE.TD_TRY_FINALIZED",),
            "加分尝试依附达阵，不能作为独立语义事件。",
        ),
        _review(
            "NFL.EVENT.PASS",
            "sha256:f6f2e00a139060409a512b85ca8f931f30bf4b6714284cef2804230b7c9b581a",
            ReviewDecision.REJECT,
            "进攻方",
            "未定义",
            "泛化 play type 不进入独立候选。",
            ("NFL.EVENT.SACK", "NFL.SCORE.PASS_TD"),
            "泛化传球混合结果，和更具体事件重叠。",
        ),
        _review(
            "NFL.EVENT.RUSH",
            "sha256:7f1ec4978947d2578553c322d81fe7f282000b850b5e298917afb5e31e3c66f3",
            ReviewDecision.REJECT,
            "进攻方",
            "未定义",
            "泛化 play type 不进入独立候选。",
            ("NFL.SCORE.RUSH_TD",),
            "泛化冲球混合结果，和更具体事件重叠。",
        ),
        _review(
            "NFL.SPECIAL.PUNT",
            "sha256:2ee8c246df1939f711d3445cccb39f5be5b5270225ea0cc82d956161bfd271ca",
            ReviewDecision.REJECT,
            "踢球队",
            "未定义",
            "泛化 special-team play 不进入独立候选。",
            ("NFL.SPECIAL.PUNT_POSITIVE_RETURN", "NFL.SPECIAL.PUNT_TOUCHBACK"),
            "泛化弃踢混合落点与回攻结果，不能独立解释。",
        ),
        _review(
            "NFL.EVENT.SUCCESS",
            "sha256:405611d5753b2eceda2b6362f818e5b7e0eea5b975da485cf861f3234ca8cf5a",
            ReviewDecision.REJECT,
            "进攻方",
            "未定义",
            "模型定义结果不进入独立候选。",
            ("NFL.EVENT.FIRST_DOWN",),
            "success 是建模标签，不是独立足球事件。",
        ),
        _review(
            "NFL.COMBO.DISTANCE_GT_10",
            "sha256:fa7f58ffb730f00b182c82db480f5733e26545664b8e3f95409299e4418e9e34",
            ReviewDecision.REJECT,
            "持球进攻方",
            "未定义",
            "状态变量不进入独立候选。",
            ("NFL.STATE.DISTANCE_LONG",),
            "距离阈值与距离状态重复，不能重复计数。",
        ),
        _review(
            "NFL.STATE.DISTANCE_LONG",
            "sha256:6fa823a73158b39bbefe887b54e4bfa8a7a2e9349a1082e9b041ca7952b0a8f0",
            ReviewDecision.REJECT,
            "持球进攻方",
            "未定义",
            "状态变量不进入独立候选。",
            ("NFL.COMBO.DISTANCE_GT_10",),
            "距离状态应作为控制项，不能与阈值重复候选。",
        ),
        _review(
            "NFL.STATE.QUARTER_4",
            "sha256:2cb0b65ac76249a55f08de038f9273cc8e6f5a2e63190ca8ab1b00b37c3a1a96",
            ReviewDecision.REJECT,
            "比赛环境",
            "未定义",
            "状态变量不进入独立候选。",
            ("NFL.STATE.GAME_TIME_FINAL_FIVE",),
            "单独节次没有动作语义，只能作为控制项。",
        ),
        _review(
            "NFL.STATE.GAME_TIME_FINAL_FIVE",
            "sha256:70175267c848990f72a67059df53d559228b355e70250def560ffae2de032b18",
            ReviewDecision.REJECT,
            "比赛环境",
            "未定义",
            "状态变量不进入独立候选。",
            ("NFL.STATE.QUARTER_4",),
            "比赛时间是条件，不是独立足球事件。",
        ),
        _review(
            "NFL.STATE.FIELD_RED_ZONE",
            "sha256:2a0891773b79abad985fd8305ae59a9af257f8f9df8c5a0d85193a05810093c7",
            ReviewDecision.REJECT,
            "持球进攻方",
            "未定义",
            "状态变量不进入独立候选。",
            ("NFL.SEQUENCE.RED_ZONE_EMPTY_DRIVE",),
            "红区位置是条件，不能替代完整回合结果。",
        ),
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_data_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_data_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_data_value(child) for child in value]
    if isinstance(value, set):
        return sorted(_canonical_data_value(child) for child in value)
    if isinstance(value, bool) or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise SportsExpertReviewError(
                "canonical evidence cannot contain non-finite numbers"
            )
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _canonical_data_value(item())
    raise SportsExpertReviewError(
        f"canonical evidence contains unsupported {type(value).__name__}"
    )


def canonical_x13_discovery_sha256(discovery_results: pd.DataFrame) -> str:
    """Content identity for the exact discovery frame passed to the freeze gate."""

    if not isinstance(discovery_results, pd.DataFrame):
        raise SportsExpertReviewError("discovery_results must be a DataFrame")
    columns = tuple(sorted(str(column) for column in discovery_results.columns))
    if len(set(columns)) != len(columns):
        raise SportsExpertReviewError("discovery_results columns must be unique")
    rows = [
        _canonical_data_value(row)
        for row in discovery_results.loc[:, list(columns)].to_dict("records")
    ]
    return _canonical_sha256(
        {
            "schema": "nfl_x13_discovery_frame_v1",
            "columns": list(columns),
            "rows": sorted(rows, key=_canonical_bytes),
        }
    )


def canonical_x13_factor_registry_sha256(
    factor_registry: Sequence[Mapping[str, object]],
) -> str:
    """Content identity for the normalized factor registry sent to the gate."""

    if not isinstance(factor_registry, Sequence) or not factor_registry:
        raise SportsExpertReviewError("factor_registry must be a non-empty sequence")
    normalized: list[object] = []
    factor_ids: set[str] = set()
    for definition in factor_registry:
        if not isinstance(definition, Mapping):
            raise SportsExpertReviewError("factor_registry definitions must be mappings")
        factor_id = _factor_id(definition.get("factor_id"), field="factor_id")
        if factor_id in factor_ids:
            raise SportsExpertReviewError("factor_registry factor_id must be unique")
        factor_ids.add(factor_id)
        normalized.append(_canonical_data_value(dict(definition)))
    return _canonical_sha256(
        {
            "schema": "nfl_x13_factor_registry_snapshot_v1",
            "factor_definitions": sorted(
                normalized,
                key=lambda item: str(dict(item)["factor_id"]),
            ),
        }
    )


def _verify_content_addressed_artifact_sha256(path: Path) -> str:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise SportsExpertReviewError("matched artifact must be a regular file")
    try:
        observed = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise SportsExpertReviewError("matched artifact is unreadable") from error
    hexadecimal = observed.removeprefix("sha256:")
    if (
        path.parent.name != f"sha256-{hexadecimal}"
        or not path.name.startswith(f"{hexadecimal}.")
    ):
        raise SportsExpertReviewError(
            "matched artifact path differs from its content address"
        )
    return observed


def _anomaly_case_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise SportsExpertReviewError("anomaly_case_ids must be a tuple")
    parsed = tuple(_nonempty_text(item, field="anomaly_case_ids") for item in value)
    if tuple(sorted(set(parsed))) != parsed:
        raise SportsExpertReviewError(
            "anomaly_case_ids must be sorted unique text"
        )
    return parsed


def _candidate_review_material(
    *,
    experiment_id: object,
    factor_id: object,
    factor_version: object,
    market_family: object,
    horizon_seconds: object,
    decision: object,
    reviewer: object,
    reviewed_at_utc: object,
    sports_rationale: object,
    anomaly_case_ids: object,
    priority: object,
    discovery_sha256: object,
    factor_registry_sha256: object,
    matched_sha256: object,
) -> dict[str, object]:
    if experiment_id != "X-13":
        raise SportsExpertReviewError("candidate review experiment_id must be X-13")
    parsed_horizon = horizon_seconds
    if type(parsed_horizon) is not int or parsed_horizon not in _REVIEW_HORIZONS:
        raise SportsExpertReviewError("candidate review horizon_seconds is invalid")
    if decision not in _REVIEW_DECISIONS:
        raise SportsExpertReviewError("candidate review decision is invalid")
    timestamp = _nonempty_text(reviewed_at_utc, field="reviewed_at_utc")
    if _UTC_TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise SportsExpertReviewError(
            "candidate review reviewed_at_utc must be a UTC timestamp"
        )
    return {
        "schema": _CANDIDATE_REVIEW_SCHEMA,
        "experiment_id": "X-13",
        "factor_id": _factor_id(factor_id, field="factor_id"),
        "factor_version": _nonempty_text(factor_version, field="factor_version"),
        "market_family": _nonempty_text(market_family, field="market_family"),
        "horizon_seconds": parsed_horizon,
        "decision": decision,
        "reviewer": _nonempty_text(reviewer, field="reviewer"),
        "reviewed_at_utc": timestamp,
        "sports_rationale": _nonempty_text(
            sports_rationale,
            field="sports_rationale",
        ),
        "anomaly_case_ids": list(_anomaly_case_ids(anomaly_case_ids)),
        "priority": _nonempty_text(priority, field="priority"),
        "discovery_sha256": _digest(
            discovery_sha256,
            field="discovery_sha256",
        ),
        "factor_registry_sha256": _digest(
            factor_registry_sha256,
            field="factor_registry_sha256",
        ),
        "matched_sha256": _digest(matched_sha256, field="matched_sha256"),
    }


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise SportsExpertReviewError(
                "candidate review artifact conflicts with content address"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise SportsExpertReviewError(
                    "candidate review artifact publish race conflicts"
                )
    finally:
        temporary.unlink(missing_ok=True)


def publish_x13_candidate_expert_review_artifact(
    *,
    output_root: Path,
    experiment_id: str,
    factor_id: str,
    factor_version: str,
    market_family: str,
    horizon_seconds: int,
    decision: str,
    reviewer: str,
    reviewed_at_utc: str,
    sports_rationale: str,
    anomaly_case_ids: tuple[str, ...],
    priority: str,
    discovery_sha256: str,
    factor_registry_sha256: str,
    matched_sha256: str,
) -> CandidateExpertReviewArtifactV1:
    """Publish one human-authored candidate review as an immutable artifact."""

    if not isinstance(output_root, Path):
        raise SportsExpertReviewError("output_root must be a Path")
    material = _candidate_review_material(
        experiment_id=experiment_id,
        factor_id=factor_id,
        factor_version=factor_version,
        market_family=market_family,
        horizon_seconds=horizon_seconds,
        decision=decision,
        reviewer=reviewer,
        reviewed_at_utc=reviewed_at_utc,
        sports_rationale=sports_rationale,
        anomaly_case_ids=anomaly_case_ids,
        priority=priority,
        discovery_sha256=discovery_sha256,
        factor_registry_sha256=factor_registry_sha256,
        matched_sha256=matched_sha256,
    )
    artifact_sha256 = _canonical_sha256(material)
    manifest = {**material, "artifact_sha256": artifact_sha256}
    payload = _canonical_bytes(manifest)
    hexadecimal = artifact_sha256.removeprefix("sha256:")
    artifact_path = (
        output_root
        / f"sha256-{hexadecimal}"
        / f"{hexadecimal}.candidate-expert-review.json"
    )
    _atomic_publish(artifact_path, payload)
    return CandidateExpertReviewArtifactV1(
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
    )


def verify_x13_candidate_expert_review_artifact(
    path: Path,
) -> VerifiedCandidateExpertReviewV1:
    """Verify artifact bytes, content address, and all human review fields."""

    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise SportsExpertReviewError("candidate review artifact must be a file")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SportsExpertReviewError(
            "candidate review artifact is unreadable"
        ) from error
    if not isinstance(value, Mapping):
        raise SportsExpertReviewError("candidate review artifact must be an object")
    artifact_sha256 = _digest(
        value.get("artifact_sha256"),
        field="artifact_sha256",
    )
    material = {key: child for key, child in value.items() if key != "artifact_sha256"}
    expected_fields = {
        "schema",
        "experiment_id",
        "factor_id",
        "factor_version",
        "market_family",
        "horizon_seconds",
        "decision",
        "reviewer",
        "reviewed_at_utc",
        "sports_rationale",
        "anomaly_case_ids",
        "priority",
        "discovery_sha256",
        "factor_registry_sha256",
        "matched_sha256",
    }
    if set(material) != expected_fields:
        raise SportsExpertReviewError("candidate review artifact fields are invalid")
    if material["schema"] != _CANDIDATE_REVIEW_SCHEMA:
        raise SportsExpertReviewError("candidate review artifact schema is invalid")
    expected_sha256 = _canonical_sha256(material)
    if artifact_sha256 != expected_sha256:
        raise SportsExpertReviewError("candidate review artifact hash differs from content")
    hexadecimal = artifact_sha256.removeprefix("sha256:")
    if (
        path.parent.name != f"sha256-{hexadecimal}"
        or path.name != f"{hexadecimal}.candidate-expert-review.json"
    ):
        raise SportsExpertReviewError(
            "candidate review artifact path differs from content address"
        )
    anomalies = material["anomaly_case_ids"]
    if not isinstance(anomalies, list):
        raise SportsExpertReviewError("anomaly_case_ids must be a JSON list")
    validated = _candidate_review_material(
        experiment_id=material["experiment_id"],
        factor_id=material["factor_id"],
        factor_version=material["factor_version"],
        market_family=material["market_family"],
        horizon_seconds=material["horizon_seconds"],
        decision=material["decision"],
        reviewer=material["reviewer"],
        reviewed_at_utc=material["reviewed_at_utc"],
        sports_rationale=material["sports_rationale"],
        anomaly_case_ids=tuple(anomalies),
        priority=material["priority"],
        discovery_sha256=material["discovery_sha256"],
        factor_registry_sha256=material["factor_registry_sha256"],
        matched_sha256=material["matched_sha256"],
    )
    return VerifiedCandidateExpertReviewV1(
        artifact_path=path,
        artifact_sha256=artifact_sha256,
        experiment_id="X-13",
        factor_id=str(validated["factor_id"]),
        factor_version=str(validated["factor_version"]),
        market_family=str(validated["market_family"]),
        horizon_seconds=int(validated["horizon_seconds"]),
        decision=str(validated["decision"]),
        reviewer=str(validated["reviewer"]),
        reviewed_at_utc=str(validated["reviewed_at_utc"]),
        sports_rationale=str(validated["sports_rationale"]),
        anomaly_case_ids=tuple(validated["anomaly_case_ids"]),
        priority=str(validated["priority"]),
        discovery_sha256=str(validated["discovery_sha256"]),
        factor_registry_sha256=str(validated["factor_registry_sha256"]),
        matched_sha256=str(validated["matched_sha256"]),
    )


def freeze_verified_x13_expert_shortlist(
    *,
    review_artifact_paths: Sequence[Path],
    discovery_results: pd.DataFrame,
    factor_registry: Sequence[Mapping[str, object]],
    matched_artifact_path: Path,
    output_root: Path,
) -> ShortlistLockV1:
    """Bind real input artifacts before calling the existing freeze gate."""

    expected_discovery = canonical_x13_discovery_sha256(discovery_results)
    expected_registry = canonical_x13_factor_registry_sha256(factor_registry)
    expected_matched = _verify_content_addressed_artifact_sha256(
        matched_artifact_path
    )
    if (
        not isinstance(review_artifact_paths, Sequence)
        or isinstance(review_artifact_paths, (str, bytes, bytearray))
        or not review_artifact_paths
    ):
        raise SportsExpertReviewError("review_artifact_paths must be non-empty")
    verified = tuple(
        verify_x13_candidate_expert_review_artifact(path)
        for path in review_artifact_paths
    )
    if any(
        review.discovery_sha256 != expected_discovery
        for review in verified
    ):
        raise SportsExpertReviewError(
            "candidate review discovery_sha256 differs from requested artifact"
        )
    if any(
        review.factor_registry_sha256 != expected_registry
        for review in verified
    ):
        raise SportsExpertReviewError(
            "candidate review factor_registry_sha256 differs from requested artifact"
        )
    if any(review.matched_sha256 != expected_matched for review in verified):
        raise SportsExpertReviewError(
            "candidate review matched_sha256 differs from requested artifact"
        )
    if not isinstance(output_root, Path):
        raise SportsExpertReviewError("output_root must be a Path")
    return _freeze_factor_shortlist(
        discovery_results=discovery_results,
        factor_registry=factor_registry,
        reviews=[review.to_human_factor_review() for review in verified],
        source_hashes=[expected_discovery, expected_registry, expected_matched],
        output_root=output_root,
    )


__all__ = [
    "CandidateExpertReviewArtifactV1",
    "ExpertReviewRecord",
    "FACTOR_REGISTRY_V2_SHA256",
    "CANONICAL_FACTOR_REGISTRY_V2_PATH",
    "ReviewDecision",
    "ReviewStatus",
    "SportsExpertReviewError",
    "VerifiedCandidateExpertReviewV1",
    "build_x13_sports_expert_review_payload",
    "canonical_payload_sha256",
    "canonical_x13_discovery_sha256",
    "canonical_x13_factor_registry_sha256",
    "freeze_verified_x13_expert_shortlist",
    "load_x13_frozen_factor_registry",
    "publish_x13_candidate_expert_review_artifact",
    "seed_x13_sports_expert_reviews",
    "verify_x13_candidate_expert_review_artifact",
    "X13_EXPERT_REVIEW_ALLOWED_CONTROL_FIELDS",
]
