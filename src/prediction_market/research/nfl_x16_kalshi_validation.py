"""Source-only Polymarket-to-Kalshi transport and sealed-holdout access ledger."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


_SOURCE_REQUIRED = {
    "game_id",
    "nfl_week",
    "cohort",
    "authority_sha256",
    "venue",
    "raw_probability",
    "outcome",
}
_TARGET_REQUIRED = _SOURCE_REQUIRED.difference({"outcome"})


class KalshiValidationError(ValueError):
    """Rows or access evidence violate the sealed X-16 validation contract."""


@dataclass(frozen=True, slots=True)
class PolymarketSourceCalibrator:
    authority_sha256: str
    coefficient: float
    intercept: float
    source_rows_sha256: str


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_authority(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise KalshiValidationError("authority_sha256 must be a sha256 digest")
    return value


def _validate_partition(
    frame: pd.DataFrame,
    *,
    required: set[str],
    cohort: str,
    venue: str,
    authority_sha256: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise KalshiValidationError("transport rows must be a nonempty DataFrame")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KalshiValidationError(f"transport rows missing required columns: {missing}")
    work = frame.copy()
    if not work["cohort"].eq(cohort).all() or not work["venue"].eq(venue).all():
        raise KalshiValidationError(f"transport requires {cohort} {venue} rows only")
    if not work["authority_sha256"].eq(authority_sha256).all():
        raise KalshiValidationError("authority_sha256 does not match the frozen assignment")
    weeks = pd.to_numeric(work["nfl_week"], errors="coerce")
    lower, upper = (1, 12) if cohort == "development" else (13, 18)
    if weeks.isna().any() or not np.equal(weeks.to_numpy(), weeks.astype(int).to_numpy()).all() or not weeks.between(lower, upper).all():
        raise KalshiValidationError(f"{cohort} nfl_week must be integral in {lower}..{upper}")
    probability = pd.to_numeric(work["raw_probability"], errors="coerce")
    if probability.isna().any() or not np.isfinite(probability.to_numpy(dtype=float)).all() or not probability.between(0, 1, inclusive="both").all():
        raise KalshiValidationError("raw_probability must be finite in [0, 1]")
    for column in ("game_id",):
        if not work[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise KalshiValidationError(f"{column} must be a nonempty string")
    return work


def fit_polymarket_source_calibrator(
    source_rows: pd.DataFrame,
    *,
    authority_sha256: str,
) -> PolymarketSourceCalibrator:
    """Fit Platt calibration from development Polymarket rows only."""

    authority = _require_authority(authority_sha256)
    source = _validate_partition(
        source_rows,
        required=_SOURCE_REQUIRED,
        cohort="development",
        venue="polymarket",
        authority_sha256=authority,
    )
    outcomes = pd.to_numeric(source["outcome"], errors="coerce")
    if outcomes.isna().any() or not outcomes.isin((0, 1)).all() or outcomes.nunique() != 2:
        raise KalshiValidationError("source outcome must be binary with both classes")
    clipped = np.clip(source["raw_probability"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", random_state=0)
    model.fit(logits, outcomes.to_numpy(dtype=int))
    provenance = source.loc[:, sorted(_SOURCE_REQUIRED)].sort_values(
        ["game_id", "nfl_week"], kind="mergesort"
    ).to_dict("records")
    return PolymarketSourceCalibrator(
        authority_sha256=authority,
        coefficient=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
        source_rows_sha256=_canonical_sha256(provenance),
    )


def transport_to_kalshi_holdout(
    target_rows: pd.DataFrame,
    *,
    calibrator: PolymarketSourceCalibrator,
    authority_sha256: str,
) -> pd.DataFrame:
    """Apply a frozen source-only calibrator to unlabeled Kalshi holdout rows."""

    authority = _require_authority(authority_sha256)
    if not isinstance(calibrator, PolymarketSourceCalibrator) or calibrator.authority_sha256 != authority:
        raise KalshiValidationError("calibrator authority_sha256 does not match the frozen assignment")
    if "outcome" in target_rows.columns:
        raise KalshiValidationError("Kalshi holdout transport must not receive outcome labels")
    target = _validate_partition(
        target_rows,
        required=_TARGET_REQUIRED,
        cohort="holdout",
        venue="kalshi",
        authority_sha256=authority,
    )
    probability = np.clip(target["raw_probability"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1 - probability))
    calibrated = 1.0 / (1.0 + np.exp(-(calibrator.intercept + calibrator.coefficient * logits)))
    result = target.copy()
    result["calibrated_probability"] = calibrated
    result["transport_source"] = "polymarket_development_only"
    result["source_calibration_sha256"] = calibrator.source_rows_sha256
    return result


def _verify_ledger(ledger: list[dict[str, object]]) -> None:
    if not isinstance(ledger, list) or not ledger:
        raise KalshiValidationError("holdout access ledger must begin with genesis")
    previous_hash: str | None = None
    count = 0
    authority: str | None = None
    for sequence, entry in enumerate(ledger):
        if not isinstance(entry, dict) or entry.get("sequence") != sequence:
            raise KalshiValidationError("holdout access ledger sequence is invalid")
        payload = {key: value for key, value in entry.items() if key != "chain_sha256"}
        if payload.get("previous_chain_sha256") != previous_hash:
            raise KalshiValidationError("holdout access ledger chain is broken")
        if entry.get("chain_sha256") != _canonical_sha256(payload):
            raise KalshiValidationError("holdout access ledger hash is invalid")
        entry_authority = _require_authority(payload.get("authority_sha256"))
        if authority is None:
            authority = entry_authority
        elif entry_authority != authority:
            raise KalshiValidationError("holdout access ledger authority changed")
        if payload.get("event") == "holdout_reaction_read":
            count += 1
        if payload.get("holdout_reaction_read_count") != count:
            raise KalshiValidationError("holdout access ledger read count is invalid")
        previous_hash = str(entry["chain_sha256"])


def begin_holdout_access_ledger(*, authority_sha256: str) -> list[dict[str, object]]:
    """Create the append-only ledger while the holdout is still unread."""

    authority = _require_authority(authority_sha256)
    entry: dict[str, object] = {
        "sequence": 0,
        "event": "genesis",
        "authority_sha256": authority,
        "previous_chain_sha256": None,
        "holdout_reaction_read_count": 0,
    }
    entry["chain_sha256"] = _canonical_sha256(entry)
    return [entry]


def append_holdout_access(
    ledger: list[dict[str, object]],
    *,
    access_kind: str,
) -> list[dict[str, object]]:
    """Append a dynamic audit event without allowing mutation of prior hashes."""

    _verify_ledger(ledger)
    if access_kind not in {"holdout_reaction_read", "metadata_read", "validation_lock"}:
        raise KalshiValidationError("unknown holdout access kind")
    prior = ledger[-1]
    count = int(prior["holdout_reaction_read_count"])
    if access_kind == "holdout_reaction_read":
        count += 1
    entry: dict[str, object] = {
        "sequence": len(ledger),
        "event": access_kind,
        "authority_sha256": prior["authority_sha256"],
        "previous_chain_sha256": prior["chain_sha256"],
        "holdout_reaction_read_count": count,
    }
    entry["chain_sha256"] = _canonical_sha256(entry)
    return [*ledger, entry]


def lock_holdout_validation(
    ledger: list[dict[str, object]],
    *,
    authority_sha256: str,
) -> dict[str, object]:
    """Issue the pre-read lock proof required before any reaction read."""

    _verify_ledger(ledger)
    authority = _require_authority(authority_sha256)
    latest = ledger[-1]
    if latest["authority_sha256"] != authority:
        raise KalshiValidationError("holdout access ledger authority does not match")
    if latest["holdout_reaction_read_count"] != 0:
        raise KalshiValidationError("holdout validation lock requires read count=0")
    return {
        "authority_sha256": authority,
        "holdout_reaction_read_count": 0,
        "chain_sha256": latest["chain_sha256"],
        "lock_event": "PRE_HOLDOUT_REACTION_READ_LOCK",
    }


__all__ = [
    "KalshiValidationError",
    "PolymarketSourceCalibrator",
    "append_holdout_access",
    "begin_holdout_access_ledger",
    "fit_polymarket_source_calibrator",
    "lock_holdout_validation",
    "transport_to_kalshi_holdout",
]
