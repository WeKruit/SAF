"""Retrospective nflfastR reference values for the sealed X-13 cohort.

The second-half receiver is reconstructed from the completed game, so these
rows are diagnostic historical evidence, never live-PIT or latency evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Protocol

import numpy as np
import pandas as pd

from prediction_market.models.nfl_fastrmodels import FEATURE_NAMES, NoSpreadModelInput


PIT_STATUS = "RETROSPECTIVE_INFERRED_NOT_LIVE_PIT"
CLAIM_BOUNDARY = "RETROSPECTIVE_DIAGNOSTIC_NOT_LIVE_PIT"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REQUIRED = {
    "game_id",
    "play_id",
    "episode_id",
    "order_sequence",
    "quarter",
    "game_seconds_remaining",
    "home_team",
    "away_team",
    "focal_team",
    "possession_team",
    "home_score",
    "away_score",
    "down",
    "distance",
    "yardline_100",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "episode_nullified",
    "episode_audit_only",
    "source_time_utc",
}


class HistoricalReferenceError(ValueError):
    """Feature rows cannot support an auditable retrospective reference."""


class HomeProbabilityPredictor(Protocol):
    def predict_home(self, model_input: NoSpreadModelInput) -> float: ...


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise HistoricalReferenceError(f"{field} must be a SHA-256 digest")
    return value


def _numeric(row: pd.Series, field: str) -> float:
    value = pd.to_numeric(pd.Series([row[field]]), errors="coerce").iloc[0]
    if pd.isna(value) or not math.isfinite(float(value)):
        raise HistoricalReferenceError(f"{field} must be finite")
    return float(value)


def _state_input(
    row: pd.Series,
    *,
    second_half_receiver: str,
) -> tuple[NoSpreadModelInput, dict[str, object]]:
    period = int(_numeric(row, "quarter"))
    posteam = str(row["possession_team"])
    home_team = str(row["home_team"])
    away_team = str(row["away_team"])
    if posteam not in {home_team, away_team}:
        raise HistoricalReferenceError("possession team is not a game participant")
    game_seconds = _numeric(row, "game_seconds_remaining")
    half_seconds = game_seconds - 1800.0 if period <= 2 else game_seconds
    home_score = _numeric(row, "home_score")
    away_score = _numeric(row, "away_score")
    score_differential = (
        home_score - away_score if posteam == home_team else away_score - home_score
    )
    elapsed_share = (3600.0 - game_seconds) / 3600.0
    diff_time_ratio = score_differential / math.exp(-4.0 * elapsed_share)
    posteam_timeouts = _numeric(
        row,
        "home_timeouts_remaining"
        if posteam == home_team
        else "away_timeouts_remaining",
    )
    defteam_timeouts = _numeric(
        row,
        "away_timeouts_remaining"
        if posteam == home_team
        else "home_timeouts_remaining",
    )
    values = (
        float(period <= 2 and posteam == second_half_receiver),
        float(posteam == home_team),
        half_seconds,
        game_seconds,
        diff_time_ratio,
        score_differential,
        _numeric(row, "down"),
        _numeric(row, "distance"),
        _numeric(row, "yardline_100"),
        posteam_timeouts,
        defteam_timeouts,
    )
    model_input = NoSpreadModelInput(
        feature_names=FEATURE_NAMES,
        feature_values=tuple(float(value) for value in values),
        posteam=posteam,
        home_team=home_team,
        away_team=away_team,
        period=period,
    )
    payload = {
        "game_id": str(row["game_id"]),
        "play_id": str(row["play_id"]),
        "episode_id": str(row["episode_id"]),
        "source_time_utc": str(row["source_time_utc"]),
        "source_row_order_sequence": int(row["order_sequence"]),
        "features": dict(zip(FEATURE_NAMES, model_input.feature_values, strict=True)),
    }
    return model_input, payload


def _valid_state(row: pd.Series) -> bool:
    quarter = pd.to_numeric(pd.Series([row["quarter"]]), errors="coerce").iloc[0]
    down = pd.to_numeric(pd.Series([row["down"]]), errors="coerce").iloc[0]
    return (
        not pd.isna(quarter)
        and 1 <= int(quarter) <= 4
        and not pd.isna(down)
        and int(down) in {1, 2, 3, 4}
        and isinstance(row["possession_team"], str)
        and bool(row["possession_team"])
    )


def build_retrospective_reference(
    feature_rows: pd.DataFrame,
    *,
    predictor: HomeProbabilityPredictor,
    model_sha256: str,
    source_object_sha256: str,
) -> pd.DataFrame:
    """Calculate before/after no-spread probabilities at episode boundaries."""

    if not isinstance(feature_rows, pd.DataFrame) or feature_rows.empty:
        raise HistoricalReferenceError("feature_rows must be non-empty")
    missing = sorted(_REQUIRED.difference(feature_rows.columns))
    if missing:
        raise HistoricalReferenceError(
            "feature_rows missing columns: " + ", ".join(missing)
        )
    model_hash = _require_sha(model_sha256, "model_sha256")
    source_hash = _require_sha(source_object_sha256, "source_object_sha256")
    rows: list[dict[str, object]] = []
    for game_id, game in feature_rows.groupby("game_id", sort=True):
        ordered = game.sort_values(
            ["order_sequence", "play_id"], kind="mergesort"
        ).reset_index(drop=True)
        q3 = ordered[
            ordered["quarter"].eq(3)
            & ordered.apply(_valid_state, axis=1)
        ]
        second_half_receiver = (
            None if q3.empty else str(q3.iloc[0]["possession_team"])
        )
        for episode_id, episode in ordered.groupby("episode_id", sort=False):
            episode_indices = episode.index.to_numpy(dtype=int)
            pre = episode.iloc[0]
            focal_team = str(pre["focal_team"])
            base = {
                "game_id": str(game_id),
                "episode_id": str(episode_id),
                "home_team": str(pre["home_team"]),
                "away_team": str(pre["away_team"]),
                "focal_team": focal_team,
                "quarter": int(pre["quarter"]),
                "model_sha256": model_hash,
                "source_object_sha256": source_hash,
                "second_half_receiver": second_half_receiver,
                "pit_status": PIT_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
            reason: str | None = None
            if int(pre["quarter"]) > 4:
                reason = "overtime"
            elif bool(episode["episode_nullified"].any()):
                reason = "nullified"
            elif bool(episode["episode_audit_only"].any()):
                reason = "audit_only"
            elif not _valid_state(pre):
                reason = "invalid_pre_state"
            elif second_half_receiver is None:
                reason = "missing_second_half_receiver"
            post: pd.Series | None = None
            if reason is None:
                later = ordered.iloc[int(episode_indices.max()) + 1 :]
                later = later[
                    ~later["episode_id"].eq(str(episode_id))
                    & later.apply(_valid_state, axis=1)
                ]
                if later.empty:
                    reason = "missing_post_state"
                else:
                    post = later.iloc[0]
            if reason is not None:
                rows.append(
                    {
                        **base,
                        "support_status": (
                            "MODEL_SUPPORT_UNPROVEN"
                            if reason == "overtime"
                            else "EXCLUDED"
                        ),
                        "exclusion_reason": reason,
                        "pre_feature_payload": None,
                        "post_feature_payload": None,
                        "pre_feature_sha256": None,
                        "post_feature_sha256": None,
                        "p_before_ref": np.nan,
                        "p_after_ref": np.nan,
                        "reference_delta": np.nan,
                    }
                )
                continue
            assert post is not None and second_half_receiver is not None
            pre_input, pre_payload = _state_input(
                pre, second_half_receiver=second_half_receiver
            )
            post_input, post_payload = _state_input(
                post, second_half_receiver=second_half_receiver
            )
            p_before_home = float(predictor.predict_home(pre_input))
            p_after_home = float(predictor.predict_home(post_input))
            if not (
                math.isfinite(p_before_home)
                and math.isfinite(p_after_home)
                and 0 <= p_before_home <= 1
                and 0 <= p_after_home <= 1
            ):
                raise HistoricalReferenceError("predictor returned invalid probability")
            home_team = str(pre["home_team"])
            p_before = (
                p_before_home if focal_team == home_team else 1.0 - p_before_home
            )
            p_after = (
                p_after_home if focal_team == home_team else 1.0 - p_after_home
            )
            rows.append(
                {
                    **base,
                    "support_status": "SUPPORTED_RETROSPECTIVE",
                    "exclusion_reason": None,
                    "pre_feature_payload": json.dumps(
                        pre_payload, sort_keys=True, separators=(",", ":")
                    ),
                    "post_feature_payload": json.dumps(
                        post_payload, sort_keys=True, separators=(",", ":")
                    ),
                    "pre_feature_sha256": _canonical_sha(pre_payload),
                    "post_feature_sha256": _canonical_sha(post_payload),
                    "p_before_ref": p_before,
                    "p_after_ref": p_after,
                    "reference_delta": p_after - p_before,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["game_id", "episode_id"], kind="mergesort"
    ).reset_index(drop=True)


def join_reference_to_market_paths(
    market_paths: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    materiality_threshold: float = 0.01,
) -> pd.DataFrame:
    """Join actual historical market changes to the focal-team reference."""

    if not 0 < materiality_threshold <= 1:
        raise HistoricalReferenceError("materiality_threshold must be in (0, 1]")
    required_market = {
        "observation_id",
        "factor_id",
        "game_id",
        "episode_id",
        "outcome",
        "venue",
        "market_family",
        "horizon_seconds",
        "effect",
    }
    missing = sorted(required_market.difference(market_paths.columns))
    if missing:
        raise HistoricalReferenceError(
            "market_paths missing columns: " + ", ".join(missing)
        )
    reference_columns = [
        "game_id",
        "episode_id",
        "focal_team",
        "p_before_ref",
        "p_after_ref",
        "reference_delta",
        "support_status",
        "pit_status",
        "model_sha256",
        "source_object_sha256",
        "pre_feature_sha256",
        "post_feature_sha256",
        "claim_boundary",
    ]
    joined = market_paths.merge(
        reference[reference_columns],
        on=["game_id", "episode_id"],
        how="left",
        validate="many_to_one",
    )
    joined["reference_orientation_valid"] = joined["outcome"].eq(
        joined["focal_team"]
    )
    supported = (
        joined["support_status"].eq("SUPPORTED_RETROSPECTIVE")
        & joined["reference_orientation_valid"]
        & joined["reference_delta"].notna()
    )
    material = supported & joined["reference_delta"].abs().ge(
        materiality_threshold
    )
    joined["reference_gap"] = np.where(
        supported,
        joined["effect"] - joined["reference_delta"],
        np.nan,
    )
    joined["completion_fraction"] = np.where(
        material,
        joined["effect"] / joined["reference_delta"],
        np.nan,
    )
    joined["completion_status"] = np.select(
        [
            ~joined["reference_orientation_valid"],
            ~joined["support_status"].eq("SUPPORTED_RETROSPECTIVE"),
            supported & ~material,
            material,
        ],
        [
            "ORIENTATION_MISMATCH",
            "REFERENCE_UNSUPPORTED",
            "BELOW_MATERIALITY",
            "ELIGIBLE",
        ],
        default="REFERENCE_MISSING",
    )
    return joined


def build_reference_summary(reference_paths: pd.DataFrame) -> pd.DataFrame:
    """Summarize market completion relative to material reference changes."""

    required = {
        "factor_id",
        "game_id",
        "venue",
        "market_family",
        "horizon_seconds",
        "effect",
        "reference_delta",
        "reference_gap",
        "completion_fraction",
        "completion_status",
    }
    missing = sorted(required.difference(reference_paths.columns))
    if missing:
        raise HistoricalReferenceError(
            "reference_paths missing columns: " + ", ".join(missing)
        )
    eligible = reference_paths[
        reference_paths["completion_status"].eq("ELIGIBLE")
    ].copy()
    rows: list[dict[str, object]] = []
    keys = ["factor_id", "venue", "market_family", "horizon_seconds"]
    for key, child in eligible.groupby(keys, sort=True, dropna=False):
        completion = child["completion_fraction"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "observed_n": int(len(child)),
                "game_count": int(child["game_id"].nunique()),
                "reference_delta_mean": float(child["reference_delta"].mean()),
                "reference_delta_median": float(child["reference_delta"].median()),
                "market_delta_mean": float(child["effect"].mean()),
                "market_delta_median": float(child["effect"].median()),
                "reference_gap_mean": float(child["reference_gap"].mean()),
                "reference_gap_median": float(child["reference_gap"].median()),
                "completion_fraction_mean": float(np.mean(completion)),
                "completion_fraction_median": float(np.median(completion)),
                "completion_fraction_p25": float(np.quantile(completion, 0.25)),
                "completion_fraction_p75": float(np.quantile(completion, 0.75)),
                "p_opposite_direction": float(np.mean(completion < 0)),
                "p_partial_0_to_1": float(
                    np.mean((completion >= 0) & (completion < 1))
                ),
                "p_near_reference": float(
                    np.mean((completion >= 0.9) & (completion <= 1.1))
                ),
                "p_overshoot_gt_1": float(np.mean(completion > 1)),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "CLAIM_BOUNDARY",
    "HistoricalReferenceError",
    "PIT_STATUS",
    "build_reference_summary",
    "build_retrospective_reference",
    "join_reference_to_market_paths",
]
