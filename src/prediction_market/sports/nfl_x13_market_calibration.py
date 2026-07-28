"""X-13 development-only NFL winner-market calibration.

The unit being calibrated is one pre-event actual trade forecast for one
finalized episode/outcome/venue.  Horizon copies are removed before any
metric is computed, and uncertainty is resampled by game rather than by
trade or episode.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = "nfl_x13_market_calibration_publication_v1"
BUILDER_VERSION = "nfl-x13-market-calibration-v1"
CLAIM_BOUNDARY = (
    "RETROSPECTIVE development-only winner-market calibration; game-clustered "
    "uncertainty; no causality, live latency, execution, tradeability, or "
    "holdout claim."
)
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FORECAST_KEY = [
    "game_id",
    "episode_id",
    "logical_market_id",
    "outcome",
    "venue",
    "pre_event_observation_id",
]
_EXACT_MARKET_BATCH_FILE_SHA = (
    "sha256:b21640b8a50bd92e2f7ed3dac07e641059f8fba9375c1dce0a47a881d655e341"
)
_EXACT_DENSE_MANIFEST_FILE_SHA = (
    "sha256:a2aa20c764523d9c117e0087eb319cc50bbf736924009c9f7b8850fb471a3ea4"
)
_EXACT_FEATURE_BATCH_FILE_SHA = (
    "sha256:3a09538aa3013d4ddab9aa5e24299c91914b98a41701854037c353326ccf4866"
)
_SPORTS_CLEAN_BATCH_FILE_SHA = (
    "sha256:3fee02261ae6aeeccbfaa03114ebdbdfd44e96eb7dad1501de5f39c2bce0271b"
)


class CalibrationPublicationError(RuntimeError):
    """A calibration input or publication failed closed."""


@dataclass(frozen=True, slots=True)
class CalibrationPublication:
    manifest_path: Path
    manifest_sha256: str
    bundle_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _inside(root: Path, relative: object, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise CalibrationPublicationError(f"{field} path is invalid")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise CalibrationPublicationError(f"{field} path escapes root") from error
    return candidate


def _verified_json(
    path: Path,
    *,
    expected_sha256: object | None,
    field: str,
) -> dict[str, object]:
    if not path.is_file():
        raise CalibrationPublicationError(f"{field} is missing")
    payload = path.read_bytes()
    actual = _sha(payload)
    if expected_sha256 is not None and actual != expected_sha256:
        raise CalibrationPublicationError(f"{field} SHA-256 mismatch")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise CalibrationPublicationError(f"{field} must be a JSON object")
    return value


def _verified_parquet(
    root: Path,
    reference: object,
    *,
    field: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if not isinstance(reference, dict):
        raise CalibrationPublicationError(f"{field} reference is invalid")
    required = {"object_path", "object_sha256", "byte_length", "row_count"}
    if not required.issubset(reference):
        raise CalibrationPublicationError(f"{field} reference is incomplete")
    path = _inside(root, reference["object_path"], field=field)
    if (
        not path.is_file()
        or path.stat().st_size != reference["byte_length"]
        or _file_sha(path) != reference["object_sha256"]
    ):
        raise CalibrationPublicationError(f"{field} object verification failed")
    table = pq.read_table(path, columns=columns)
    if table.num_rows != reference["row_count"]:
        raise CalibrationPublicationError(f"{field} row count mismatch")
    return table.to_pandas()


def _score_table(
    feature_rows: pd.DataFrame,
    *,
    feature_object_sha256s: Mapping[str, str],
) -> pd.DataFrame:
    required = {"game_id", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(feature_rows.columns)
    if missing:
        raise CalibrationPublicationError(
            f"feature rows lack final-score fields: {sorted(missing)}"
        )
    rows: list[dict[str, object]] = []
    for game_id, group in feature_rows.groupby("game_id", sort=True):
        home_teams = {
            str(value).strip()
            for value in group["home_team"].dropna().tolist()
            if str(value).strip()
        }
        away_teams = {
            str(value).strip()
            for value in group["away_team"].dropna().tolist()
            if str(value).strip()
        }
        if len(home_teams) != 1 or len(away_teams) != 1:
            raise CalibrationPublicationError(f"{game_id} team identity is ambiguous")
        home_team = next(iter(home_teams))
        away_team = next(iter(away_teams))
        if home_team == away_team:
            raise CalibrationPublicationError(f"{game_id} home and away teams collide")
        home_scores = pd.to_numeric(group["home_score"], errors="coerce").dropna()
        away_scores = pd.to_numeric(group["away_score"], errors="coerce").dropna()
        if home_scores.empty or away_scores.empty:
            raise CalibrationPublicationError(f"{game_id} final score is unavailable")
        source_sha = feature_object_sha256s.get(str(game_id))
        if not isinstance(source_sha, str) or not _SHA_RE.fullmatch(source_sha):
            raise CalibrationPublicationError(
                f"{game_id} feature object SHA-256 is unavailable"
            )
        rows.append(
            {
                "game_id": str(game_id),
                "home_team": home_team,
                "away_team": away_team,
                # Scores are non-decreasing state variables.  The maximum in the
                # verified feature object is therefore its terminal observed score.
                "final_home_score": int(home_scores.max()),
                "final_away_score": int(away_scores.max()),
                "final_score_source_object_sha256": source_sha,
                "final_score_verification": (
                    "VERIFIED_CANONICAL_EVENT_OBJECT_MONOTONE_TERMINAL_MAX"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_forecast_observations(
    *,
    reaction_paths: pd.DataFrame,
    feature_rows: pd.DataFrame,
    feature_object_sha256s: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct one binary forecast per unique pre-event actual trade."""

    required = {
        *_FORECAST_KEY,
        "market_family",
    }
    probability_field = (
        "pre_probability"
        if "pre_probability" in reaction_paths.columns
        else "pre_event_actual_trade"
    )
    required.add(probability_field)
    missing = required - set(reaction_paths.columns)
    if missing:
        raise CalibrationPublicationError(
            f"reaction paths lack forecast fields: {sorted(missing)}"
        )
    work = reaction_paths.loc[:, [*required]].copy()
    work = work.rename(columns={probability_field: "forecast_probability"})
    exclusion_counts: dict[str, int] = {}

    non_winner = work["market_family"].astype(str).str.lower() != "moneyline"
    exclusion_counts["NON_WINNER_MARKET_FAMILY"] = int(non_winner.sum())
    work = work.loc[~non_winner].copy()

    missing_forecast = (
        work["pre_event_observation_id"].isna()
        | work["forecast_probability"].isna()
    )
    if missing_forecast.any():
        exclusion_counts["MISSING_PRE_EVENT_ACTUAL_TRADE"] = int(
            missing_forecast.sum()
        )
        work = work.loc[~missing_forecast].copy()
    work["forecast_probability"] = pd.to_numeric(
        work["forecast_probability"], errors="coerce"
    )
    if (
        work["forecast_probability"].isna().any()
        or (work["forecast_probability"] < 0).any()
        or (work["forecast_probability"] > 1).any()
    ):
        raise CalibrationPublicationError("forecast probability is outside [0, 1]")

    consistency_fields = ["forecast_probability", "market_family"]
    inconsistent = (
        work.groupby(_FORECAST_KEY, dropna=False)[consistency_fields]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if inconsistent.any():
        raise CalibrationPublicationError(
            "horizon copies disagree on the pre-event forecast"
        )
    work = work.sort_values(_FORECAST_KEY, kind="mergesort").drop_duplicates(
        _FORECAST_KEY, keep="first"
    )

    scores = _score_table(
        feature_rows,
        feature_object_sha256s=feature_object_sha256s,
    )
    work = work.merge(scores, on="game_id", how="left", validate="many_to_one")
    if work["home_team"].isna().any():
        raise CalibrationPublicationError(
            "reaction paths include a game outside the verified feature batch"
        )

    outcome = work["outcome"].astype(str)
    home = work["home_team"].astype(str)
    away = work["away_team"].astype(str)
    unknown = ~(outcome.eq(home) | outcome.eq(away))
    if unknown.any():
        bad = sorted(
            {
                f"{game}:{team}"
                for game, team in zip(
                    work.loc[unknown, "game_id"],
                    work.loc[unknown, "outcome"],
                    strict=True,
                )
            }
        )
        raise CalibrationPublicationError(
            f"outcome orientation cannot be proved: {bad[:5]}"
        )
    work["orientation_proof"] = np.where(
        outcome.eq(home),
        "OUTCOME_EQUALS_VERIFIED_HOME_TEAM",
        "OUTCOME_EQUALS_VERIFIED_AWAY_TEAM",
    )

    tied = work["final_home_score"].eq(work["final_away_score"])
    if tied.any():
        exclusion_counts["BINARY_LABEL_UNDEFINED_TIE"] = int(tied.sum())
        work = work.loc[~tied].copy()
        outcome = work["outcome"].astype(str)
        home = work["home_team"].astype(str)
    home_won = work["final_home_score"].gt(work["final_away_score"])
    work["realized_label"] = np.where(
        outcome.eq(home),
        home_won,
        ~home_won,
    ).astype("int8")
    work["home_away_proxy"] = np.where(outcome.eq(home), "HOME", "AWAY")
    work["favorite_underdog_proxy"] = np.select(
        [
            work["forecast_probability"].gt(0.5),
            work["forecast_probability"].lt(0.5),
        ],
        ["FAVORITE_AT_EPISODE", "UNDERDOG_AT_EPISODE"],
        default="EVEN_AT_EPISODE",
    )
    work["forecast_grain"] = (
        "game+episode+logical_market+outcome+venue+pre_event_observation_id"
    )
    work["cluster_unit"] = "game_id"
    work["claim_boundary"] = CLAIM_BOUNDARY
    ordered = [
        *_FORECAST_KEY,
        "market_family",
        "forecast_probability",
        "realized_label",
        "home_team",
        "away_team",
        "final_home_score",
        "final_away_score",
        "orientation_proof",
        "final_score_verification",
        "final_score_source_object_sha256",
        "home_away_proxy",
        "favorite_underdog_proxy",
        "forecast_grain",
        "cluster_unit",
        "claim_boundary",
    ]
    work = work.loc[:, ordered].sort_values(_FORECAST_KEY, kind="mergesort")
    work = work.reset_index(drop=True)
    exclusions = pd.DataFrame(
        [
            {"reason": reason, "row_count": count}
            for reason, count in sorted(exclusion_counts.items())
            if count
        ],
        columns=["reason", "row_count"],
    )
    return work, exclusions


def _fit_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    clipped = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    y = labels.astype(float)
    sample_weight = (
        np.ones(len(y), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if sample_weight.sum() <= 0 or len(np.unique(y[sample_weight > 0])) < 2:
        return math.nan, math.nan
    intercept = 0.0
    slope = 1.0
    for _ in range(60):
        linear = np.clip(intercept + slope * logits, -40, 40)
        fitted = 1 / (1 + np.exp(-linear))
        variance = np.maximum(fitted * (1 - fitted), 1e-10) * sample_weight
        residual = (y - fitted) * sample_weight
        gradient = np.array([residual.sum(), (residual * logits).sum()])
        hessian = np.array(
            [
                [variance.sum() + 1e-8, (variance * logits).sum()],
                [
                    (variance * logits).sum(),
                    (variance * logits * logits).sum() + 1e-8,
                ],
            ]
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return math.nan, math.nan
        intercept += float(step[0])
        slope += float(step[1])
        if max(abs(float(step[0])), abs(float(step[1]))) < 1e-9:
            break
    if not np.isfinite(intercept) or not np.isfinite(slope):
        return math.nan, math.nan
    return float(slope), float(intercept)


def _point_metrics(frame: pd.DataFrame) -> dict[str, float]:
    p = frame["forecast_probability"].to_numpy(dtype=float)
    y = frame["realized_label"].to_numpy(dtype=float)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    slope, intercept = _fit_calibration(p, y)
    return {
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(
            -np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
        ),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    game_ids = sorted(frame["game_id"].unique())
    if not game_ids:
        raise CalibrationPublicationError("calibration group has no games")
    game_index = {game_id: index for index, game_id in enumerate(game_ids)}
    row_game = frame["game_id"].map(game_index).to_numpy(dtype=int)
    p = frame["forecast_probability"].to_numpy(dtype=float)
    y = frame["realized_label"].to_numpy(dtype=float)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    brier = (p - y) ** 2
    logloss = -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
    count_by_game = np.bincount(row_game, minlength=len(game_ids)).astype(float)
    brier_by_game = np.bincount(
        row_game, weights=brier, minlength=len(game_ids)
    )
    logloss_by_game = np.bincount(
        row_game, weights=logloss, minlength=len(game_ids)
    )

    # Prices live on a small venue tick grid.  Collapse rows to
    # game × probability sufficient counts, then fit bootstrap calibration in
    # vectorized batches.  This preserves exact game-cluster resampling without
    # treating the underlying episode forecasts as independent clusters.
    unique_p, p_index = np.unique(p, return_inverse=True)
    count_grid = np.zeros((len(game_ids), len(unique_p)), dtype=float)
    positive_grid = np.zeros_like(count_grid)
    np.add.at(count_grid, (row_game, p_index), 1.0)
    np.add.at(positive_grid, (row_game, p_index), y)
    calibration_logits = np.log(
        np.clip(unique_p, 1e-6, 1 - 1e-6)
        / (1 - np.clip(unique_p, 1e-6, 1 - 1e-6))
    )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    batch_size = 256
    for batch_start in range(0, samples, batch_size):
        batch_count = min(batch_size, samples - batch_start)
        sampled = rng.integers(
            0,
            len(game_ids),
            size=(batch_count, len(game_ids)),
        )
        game_weights = np.zeros(
            (batch_count, len(game_ids)), dtype=float
        )
        np.add.at(
            game_weights,
            (
                np.arange(batch_count)[:, None],
                sampled,
            ),
            1.0,
        )
        denominator = game_weights @ count_by_game
        brier_draws = game_weights @ brier_by_game / denominator
        logloss_draws = game_weights @ logloss_by_game / denominator
        counts = game_weights @ count_grid
        positives = game_weights @ positive_grid
        intercepts = np.zeros(batch_count, dtype=float)
        slopes = np.ones(batch_count, dtype=float)
        identifiable = (positives.sum(axis=1) > 0) & (
            positives.sum(axis=1) < counts.sum(axis=1)
        )
        active = identifiable.copy()
        for _ in range(60):
            linear = np.clip(
                intercepts[:, None]
                + slopes[:, None] * calibration_logits[None, :],
                -40,
                40,
            )
            fitted = 1 / (1 + np.exp(-linear))
            variance = np.maximum(
                counts * fitted * (1 - fitted),
                1e-10 * (counts > 0),
            )
            residual = positives - counts * fitted
            gradient0 = residual.sum(axis=1)
            gradient1 = (residual * calibration_logits[None, :]).sum(
                axis=1
            )
            h00 = variance.sum(axis=1) + 1e-8
            h01 = (variance * calibration_logits[None, :]).sum(axis=1)
            h11 = (
                variance * calibration_logits[None, :] ** 2
            ).sum(axis=1) + 1e-8
            determinant = h00 * h11 - h01 * h01
            valid = active & (determinant > 0) & np.isfinite(determinant)
            step0 = np.zeros(batch_count, dtype=float)
            step1 = np.zeros(batch_count, dtype=float)
            step0[valid] = (
                gradient0[valid] * h11[valid]
                - gradient1[valid] * h01[valid]
            ) / determinant[valid]
            step1[valid] = (
                h00[valid] * gradient1[valid]
                - h01[valid] * gradient0[valid]
            ) / determinant[valid]
            intercepts[valid] += step0[valid]
            slopes[valid] += step1[valid]
            converged = np.maximum(np.abs(step0), np.abs(step1)) < 1e-9
            active &= ~converged
            active &= valid
            if not active.any():
                break
        slopes[~identifiable] = np.nan
        intercepts[~identifiable] = np.nan
        for offset in range(batch_count):
            rows.append(
                {
                    "replicate": batch_start + offset,
                    "brier_score": float(brier_draws[offset]),
                    "log_loss": float(logloss_draws[offset]),
                    "calibration_slope": float(slopes[offset]),
                    "calibration_intercept": float(intercepts[offset]),
                }
            )
    return pd.DataFrame(rows)


def _ci(series: pd.Series) -> tuple[float, float, int]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, 0
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        int(len(values)),
    )


def _scopes(observations: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    groups = [("overall", "all", observations)]
    groups.extend(
        ("venue", str(venue), group.copy())
        for venue, group in observations.groupby("venue", sort=True)
    )
    return groups


def build_calibration_tables(
    observations: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    reliability_bin_count: int = 10,
) -> dict[str, pd.DataFrame]:
    """Compute metrics with game-clustered bootstrap uncertainty."""

    if bootstrap_samples < 1:
        raise CalibrationPublicationError("bootstrap_samples must be positive")
    if reliability_bin_count < 2:
        raise CalibrationPublicationError("reliability_bin_count must be >= 2")
    required = {
        "game_id",
        "venue",
        "forecast_probability",
        "realized_label",
        "home_away_proxy",
        "favorite_underdog_proxy",
    }
    missing = required - set(observations.columns)
    if missing or observations.empty:
        raise CalibrationPublicationError(
            f"forecast observations are unavailable: {sorted(missing)}"
        )
    metric_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    bootstrap_parts: list[pd.DataFrame] = []
    for scope_index, (scope, scope_value, frame) in enumerate(
        _scopes(observations)
    ):
        point = _point_metrics(frame)
        bootstrap = _cluster_bootstrap(
            frame,
            samples=bootstrap_samples,
            seed=bootstrap_seed + scope_index,
        )
        bootstrap.insert(0, "scope_value", scope_value)
        bootstrap.insert(0, "scope", scope)
        bootstrap_parts.append(bootstrap)
        cis = {
            name: _ci(bootstrap[name])
            for name in (
                "brier_score",
                "log_loss",
                "calibration_slope",
                "calibration_intercept",
            )
        }
        metric_rows.append(
            {
                "scope": scope,
                "scope_value": scope_value,
                "forecast_count": len(frame),
                "game_count": frame["game_id"].nunique(),
                "cluster_unit": "game_id",
                **point,
                "brier_ci_low": cis["brier_score"][0],
                "brier_ci_high": cis["brier_score"][1],
                "log_loss_ci_low": cis["log_loss"][0],
                "log_loss_ci_high": cis["log_loss"][1],
                "calibration_slope_ci_low": cis["calibration_slope"][0],
                "calibration_slope_ci_high": cis["calibration_slope"][1],
                "calibration_intercept_ci_low": cis[
                    "calibration_intercept"
                ][0],
                "calibration_intercept_ci_high": cis[
                    "calibration_intercept"
                ][1],
                "bootstrap_samples_requested": bootstrap_samples,
                "bootstrap_samples_valid_brier": cis["brier_score"][2],
                "bootstrap_samples_valid_log_loss": cis["log_loss"][2],
                "bootstrap_samples_valid_calibration": min(
                    cis["calibration_slope"][2],
                    cis["calibration_intercept"][2],
                ),
                "bootstrap_seed": bootstrap_seed + scope_index,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        indices = np.minimum(
            (
                frame["forecast_probability"].to_numpy(dtype=float)
                * reliability_bin_count
            ).astype(int),
            reliability_bin_count - 1,
        )
        binned = frame.assign(_bin=indices)
        for bin_index, bin_frame in binned.groupby("_bin", sort=True):
            reliability_rows.append(
                {
                    "scope": scope,
                    "scope_value": scope_value,
                    "bin_index": int(bin_index),
                    "bin_lower": int(bin_index) / reliability_bin_count,
                    "bin_upper": (int(bin_index) + 1)
                    / reliability_bin_count,
                    "forecast_count": len(bin_frame),
                    "game_count": bin_frame["game_id"].nunique(),
                    "mean_forecast": float(
                        bin_frame["forecast_probability"].mean()
                    ),
                    "realized_rate": float(
                        bin_frame["realized_label"].mean()
                    ),
                    "brier_score": float(
                        np.mean(
                            (
                                bin_frame["forecast_probability"]
                                - bin_frame["realized_label"]
                            )
                            ** 2
                        )
                    ),
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )

    bias_rows: list[dict[str, object]] = []
    for family, field in (
        ("home_away", "home_away_proxy"),
        ("favorite_underdog", "favorite_underdog_proxy"),
    ):
        for value, frame in observations.groupby(field, sort=True):
            bias_rows.append(
                {
                    "proxy_family": family,
                    "proxy_value": str(value),
                    "status": "COMPUTED_PROXY",
                    "forecast_count": len(frame),
                    "game_count": frame["game_id"].nunique(),
                    "mean_forecast": float(
                        frame["forecast_probability"].mean()
                    ),
                    "realized_rate": float(frame["realized_label"].mean()),
                    "brier_score": float(
                        np.mean(
                            (
                                frame["forecast_probability"]
                                - frame["realized_label"]
                            )
                            ** 2
                        )
                    ),
                    "claim_boundary": (
                        "DESCRIPTIVE_PROXY_ONLY_NOT_CAUSAL_BIAS_EVIDENCE"
                    ),
                }
            )
    bias_rows.append(
        {
            "proxy_family": "popularity",
            "proxy_value": "UNAVAILABLE",
            "status": "DATA_GAP",
            "forecast_count": 0,
            "game_count": 0,
            "mean_forecast": math.nan,
            "realized_rate": math.nan,
            "brier_score": math.nan,
            "claim_boundary": "NO_RELIABLE_POPULARITY_SOURCE_NOT_GUESSED",
        }
    )
    return {
        "metrics": pd.DataFrame(metric_rows),
        "reliability": pd.DataFrame(reliability_rows),
        "bootstrap": pd.concat(bootstrap_parts, ignore_index=True),
        "bias_proxies": pd.DataFrame(bias_rows),
    }


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        buffer,
        compression="zstd",
        version="2.6",
        write_statistics=True,
    )
    return buffer.getvalue()


def _publish_bytes(root: Path, payload: bytes, *, suffix: str) -> dict[str, object]:
    sha = _sha(payload)
    digest = sha.split(":", 1)[1]
    relative = Path("objects") / "sha256" / digest[:2] / f"{digest}.{suffix}"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise CalibrationPublicationError("content-address collision")
    else:
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return {
        "object_path": relative.as_posix(),
        "object_sha256": sha,
        "byte_length": len(payload),
    }


def publish_calibration(
    *,
    output_root: str | Path,
    observations: pd.DataFrame,
    exclusions: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    input_hashes: Mapping[str, str],
    expected_game_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> CalibrationPublication:
    """Publish a deterministic content-addressed calibration bundle."""

    required_inputs = {
        "market_batch_index_sha256",
        "market_batch_sha256",
        "dense_manifest_sha256",
        "dense_bundle_sha256",
        "feature_batch_index_sha256",
        "feature_batch_sha256",
        "sports_clean_batch_index_sha256",
        "sports_clean_batch_sha256",
    }
    if set(input_hashes) != required_inputs or any(
        not isinstance(value, str) or not _SHA_RE.fullmatch(value)
        for value in input_hashes.values()
    ):
        raise CalibrationPublicationError("input hash set is invalid")
    labeled_game_count = int(observations["game_id"].nunique())
    if labeled_game_count < 1 or labeled_game_count > expected_game_count:
        raise CalibrationPublicationError("development labeled game count mismatch")
    required_tables = {"metrics", "reliability", "bootstrap", "bias_proxies"}
    if set(tables) != required_tables:
        raise CalibrationPublicationError("calibration table set is incomplete")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    objects: dict[str, dict[str, object]] = {}
    frames: dict[str, pd.DataFrame] = {
        "forecast_observations": observations,
        "exclusions": exclusions,
        **dict(tables),
    }
    for name, frame in frames.items():
        payload = _parquet_bytes(frame)
        reference = _publish_bytes(root, payload, suffix="parquet")
        reference.update(
            {
                "format": "parquet",
                "row_count": len(frame),
                "column_count": len(frame.columns),
            }
        )
        objects[name] = reference
    bundle_core = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "experiment_id": "X-13",
        "cohort": "development",
        "source_game_count": expected_game_count,
        "labeled_game_count": labeled_game_count,
        "forecast_count": len(observations),
        "forecast_grain": (
            "game+episode+logical_market+outcome+venue+pre_event_observation_id"
        ),
        "cluster_unit": "game_id",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "input_hashes": dict(sorted(input_hashes.items())),
        "objects": objects,
        "holdout_reaction_access": "CLOSED",
        "holdout_reaction_rows_read": 0,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    bundle_sha = _sha(_canonical_bytes(bundle_core))
    manifest = {**bundle_core, "bundle_sha256": bundle_sha}
    payload = _canonical_bytes(manifest)
    manifest_sha = _sha(payload)
    digest = manifest_sha.split(":", 1)[1]
    relative = (
        Path("manifests")
        / "sha256"
        / digest[:2]
        / f"{digest}.manifest.json"
    )
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise CalibrationPublicationError("manifest collision")
    else:
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return CalibrationPublication(path, manifest_sha, bundle_sha)


def load_exact_development_inputs(
    *,
    market_root: str | Path,
    market_batch_index_path: str | Path,
    dense_root: str | Path,
    dense_manifest_path: str | Path,
    feature_root: str | Path,
    feature_batch_index_path: str | Path,
    sports_clean_root: str | Path,
    sports_clean_batch_index_path: str | Path,
    expected_game_count: int = 153,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Verify exact immutable development lineage; never read holdout."""

    market_root = Path(market_root).resolve()
    dense_root = Path(dense_root).resolve()
    feature_root = Path(feature_root).resolve()
    sports_clean_root = Path(sports_clean_root).resolve()
    market_batch_path = Path(market_batch_index_path).resolve()
    dense_batch_path = Path(dense_manifest_path).resolve()
    feature_batch_path = Path(feature_batch_index_path).resolve()
    sports_clean_batch_path = Path(sports_clean_batch_index_path).resolve()
    expected_files = (
        (market_batch_path, _EXACT_MARKET_BATCH_FILE_SHA, "market batch"),
        (dense_batch_path, _EXACT_DENSE_MANIFEST_FILE_SHA, "dense manifest"),
        (feature_batch_path, _EXACT_FEATURE_BATCH_FILE_SHA, "feature batch"),
    )
    for path, expected, field in expected_files:
        if _file_sha(path) != expected:
            raise CalibrationPublicationError(f"{field} is not the frozen input")
    if _file_sha(sports_clean_batch_path) != _SPORTS_CLEAN_BATCH_FILE_SHA:
        raise CalibrationPublicationError(
            "sports clean batch is not the feature-linked frozen input"
        )
    market = _verified_json(
        market_batch_path,
        expected_sha256=_EXACT_MARKET_BATCH_FILE_SHA,
        field="market batch",
    )
    dense = _verified_json(
        dense_batch_path,
        expected_sha256=_EXACT_DENSE_MANIFEST_FILE_SHA,
        field="dense manifest",
    )
    feature = _verified_json(
        feature_batch_path,
        expected_sha256=_EXACT_FEATURE_BATCH_FILE_SHA,
        field="feature batch",
    )
    sports_clean = _verified_json(
        sports_clean_batch_path,
        expected_sha256=_SPORTS_CLEAN_BATCH_FILE_SHA,
        field="sports clean batch",
    )
    if (
        market.get("game_count") != expected_game_count
        or dense.get("game_count") != expected_game_count
        or feature.get("published_game_count") != expected_game_count
    ):
        raise CalibrationPublicationError("frozen development count mismatch")
    if market.get("final_holdout_access") != "CLOSED":
        raise CalibrationPublicationError("market batch does not seal holdout")
    if (
        dense.get("final_holdout_access") != "CLOSED"
        or dense.get("holdout_reaction_accessed") is not False
    ):
        raise CalibrationPublicationError("dense batch does not seal holdout")
    if feature.get("reaction_access") != "CLOSED":
        raise CalibrationPublicationError("feature batch reaction access is open")
    if (
        sports_clean.get("reaction_access") != "CLOSED"
        or sports_clean.get("batch_sha256")
        != feature.get("sports_clean_batch_sha256")
    ):
        raise CalibrationPublicationError(
            "sports clean batch is not sealed feature provenance"
        )

    def references(batch: Mapping[str, object], field: str) -> dict[str, dict]:
        games = batch.get("games")
        if not isinstance(games, list) or len(games) != expected_game_count:
            raise CalibrationPublicationError(f"{field} game list is invalid")
        result: dict[str, dict] = {}
        for row in games:
            if not isinstance(row, dict):
                raise CalibrationPublicationError(f"{field} game ref is invalid")
            game_id = str(row.get("game_id", ""))
            if not game_id or game_id in result:
                raise CalibrationPublicationError(f"{field} game ID is invalid")
            result[game_id] = row
        return result

    market_refs = references(market, "market")
    dense_refs = references(dense, "dense")
    feature_refs = references(feature, "feature")
    sports_games = sports_clean.get("games")
    if not isinstance(sports_games, list):
        raise CalibrationPublicationError("sports clean game list is invalid")
    sports_refs = {
        str(row.get("game_id", "")): row
        for row in sports_games
        if isinstance(row, dict) and row.get("game_id")
    }
    if set(market_refs) != set(dense_refs) or set(market_refs) != set(feature_refs):
        raise CalibrationPublicationError("frozen development game sets differ")
    if not set(feature_refs).issubset(sports_refs):
        raise CalibrationPublicationError(
            "feature games are absent from sports clean provenance"
        )

    path_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    feature_object_shas: dict[str, str] = {}
    for game_id in sorted(market_refs):
        market_ref = market_refs[game_id]
        market_manifest_path = _inside(
            market_root,
            market_ref.get("manifest_path"),
            field=f"market.{game_id}",
        )
        market_manifest = _verified_json(
            market_manifest_path,
            expected_sha256=market_ref.get("manifest_sha256"),
            field=f"market.{game_id}",
        )
        stages = market_manifest.get("stages")
        if not isinstance(stages, dict):
            raise CalibrationPublicationError(f"market.{game_id} stages invalid")
        reaction = _verified_parquet(
            market_root,
            stages.get("reaction_paths"),
            field=f"market.{game_id}.reaction_paths",
            columns=[
                "game_id",
                "episode_id",
                "logical_market_id",
                "outcome",
                "venue",
                "horizon_seconds",
                "pre_event_actual_trade",
                "pre_event_observation_id",
            ],
        )
        reaction["market_family"] = "moneyline"
        path_parts.append(reaction)

        feature_ref = feature_refs[game_id]
        feature_manifest_path = _inside(
            feature_root,
            feature_ref.get("manifest_path"),
            field=f"feature.{game_id}",
        )
        feature_manifest = _verified_json(
            feature_manifest_path,
            expected_sha256=feature_ref.get("manifest_sha256"),
            field=f"feature.{game_id}",
        )
        feature_view = feature_manifest.get("feature_view")
        # Verify the declared feature object, but do not infer the final score
        # from factor-eligible rows: OT terminal scoring can occur after the
        # final factor row.  The linked canonical replay is the score source.
        _verified_parquet(
            feature_root,
            feature_view,
            field=f"feature.{game_id}.feature_view",
            columns=[
                "game_id",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
            ],
        )
        sports_ref = sports_refs[game_id]
        sports_manifest = _verified_json(
            _inside(
                sports_clean_root,
                sports_ref.get("manifest_path"),
                field=f"sports_clean.{game_id}",
            ),
            expected_sha256=sports_ref.get("manifest_sha256"),
            field=f"sports_clean.{game_id}",
        )
        feature_inputs = feature_manifest.get("inputs")
        if (
            not isinstance(feature_inputs, dict)
            or sports_manifest.get("bundle_sha256")
            != feature_inputs.get("sports_clean_bundle_sha256")
        ):
            raise CalibrationPublicationError(
                f"{game_id} feature/sports clean bundle mismatch"
            )
        sports_stages = sports_manifest.get("stages")
        if not isinstance(sports_stages, dict):
            raise CalibrationPublicationError(
                f"sports_clean.{game_id} stages invalid"
            )
        canonical_ref = sports_stages.get("canonical_events")
        canonical_scores = _verified_parquet(
            sports_clean_root,
            canonical_ref,
            field=f"sports_clean.{game_id}.canonical_events",
            columns=[
                "game_id",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
            ],
        )
        feature_parts.append(canonical_scores)
        feature_object_shas[game_id] = str(canonical_ref["object_sha256"])

        dense_ref = dense_refs[game_id]
        _verified_json(
            _inside(
                dense_root,
                dense_ref.get("manifest_path"),
                field=f"dense.{game_id}",
            ),
            expected_sha256=dense_ref.get("manifest_sha256"),
            field=f"dense.{game_id}",
        )

    input_hashes = {
        "market_batch_index_sha256": _EXACT_MARKET_BATCH_FILE_SHA,
        "market_batch_sha256": str(market["batch_sha256"]),
        "dense_manifest_sha256": _EXACT_DENSE_MANIFEST_FILE_SHA,
        "dense_bundle_sha256": str(dense["bundle_sha256"]),
        "feature_batch_index_sha256": _EXACT_FEATURE_BATCH_FILE_SHA,
        "feature_batch_sha256": str(feature["batch_sha256"]),
        "sports_clean_batch_index_sha256": _SPORTS_CLEAN_BATCH_FILE_SHA,
        "sports_clean_batch_sha256": str(sports_clean["batch_sha256"]),
    }
    return (
        pd.concat(path_parts, ignore_index=True),
        pd.concat(feature_parts, ignore_index=True),
        input_hashes | {
            f"feature_object:{game_id}": sha
            for game_id, sha in feature_object_shas.items()
        },
    )
