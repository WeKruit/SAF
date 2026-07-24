"""Verified official nflverse fastrmodels no-spread predictor.

The frozen golden vector in this module is an engineering regression fixture.
It is not an accuracy oracle and this module performs no empirical evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Final, Iterable

import numpy as np
import xgboost

from prediction_market.program_audit import (
    ResearchRegistryError,
    load_dataset_registry,
)
from prediction_market.static_store import (
    StaticStoreError,
    VerifiedStaticObject,
    read_verified_static_object,
)


DATASET_ID: Final = "DS-NFL-FASTRMODELS"
MODEL_ID: Final = "MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1"
ASSET_ID: Final = 253928623
ASSET_URL: Final = (
    "https://github.com/nflverse/fastrmodels/releases/download/"
    "model_archive/wp_model.ubj"
)
ASSET_BYTE_LENGTH: Final = 106951
ASSET_SHA256: Final = (
    "sha256:"
    "ff58c98d22e3e36f50f6e82f8e68d3ed0920101132a50d30bb70291df33c0a4c"
)
ASSET_MANIFEST_SHA256: Final = (
    "sha256:"
    "080d98f34495fe59a532b7c24e17536f471700e92ac8415b682234d7241fe3cb"
)
ARCHIVE_TAG_COMMIT: Final = "9f2495fdb4943087ca663d96706eb5df7973aff4"
FEATURE_SPEC_COMMIT: Final = "75c7b68bc49535370236c38c9826265da075bd71"
FEATURE_HELPER_COMMIT: Final = "ead5e2f9641490f692d923c04835bd3b90275b4e"
REQUIRED_XGBOOST_VERSION: Final = "3.3.0"
EXPECTED_FEATURE_COUNT: Final = 11
EXPECTED_BOOSTED_ROUNDS: Final = 65
ASSET_SCHEMA_FINGERPRINT: Final = (
    "sha256:"
    "0032e7efd41481e00519a018d5c572de559b191f790eebafcdaf1307e0942987"
)
EXPECTED_SOURCE_REQUEST = MappingProxyType(
    {
        "archive_tag_commit": ARCHIVE_TAG_COMMIT,
        "github_release_asset_id": ASSET_ID,
        "method": "GET",
    }
)
EXPECTED_SOURCE_CURSOR: Final = (
    "archive_commit="
    f"{ARCHIVE_TAG_COMMIT};asset_id={ASSET_ID}"
)
FEATURE_NAMES: Final = (
    "receive_2h_ko",
    "home",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "Diff_Time_Ratio",
    "score_differential",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
)
GOLDEN_VECTOR_EVIDENCE_CLASS: Final = (
    "runtime_regression_fixture_not_accuracy_oracle"
)


class OfficialModelAssetError(StaticStoreError):
    """The verified bytes do not match the frozen official model identity."""


class OfficialModelInputError(ValueError):
    """A feature vector is not eligible for the frozen no-spread booster."""


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OfficialModelInputError(f"{field} must be numeric and finite")
    result = float(value)
    if not math.isfinite(result):
        raise OfficialModelInputError(f"{field} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class NoSpreadModelInput:
    """One possession-oriented regulation state in the official feature order."""

    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    posteam: str
    home_team: str
    away_team: str
    period: int

    def __post_init__(self) -> None:
        if self.feature_names != FEATURE_NAMES:
            raise OfficialModelInputError(
                "feature names or order differ from the official no-spread model"
            )
        if (
            type(self.feature_values) is not tuple
            or len(self.feature_values) != EXPECTED_FEATURE_COUNT
        ):
            raise OfficialModelInputError(
                "feature values must match the official feature count"
            )
        values = tuple(
            _finite(value, f"feature_values[{index}]")
            for index, value in enumerate(self.feature_values)
        )
        object.__setattr__(self, "feature_values", values)
        teams = (self.home_team, self.away_team)
        if (
            any(
                type(team) is not str
                or not team
                or team != team.strip()
                for team in teams
            )
            or self.home_team == self.away_team
        ):
            raise OfficialModelInputError(
                "home and away teams must be distinct canonical identifiers"
            )
        if self.posteam not in teams:
            raise OfficialModelInputError(
                "possession team must be a game participant"
            )
        if (
            isinstance(self.period, bool)
            or type(self.period) is not int
            or self.period not in {1, 2, 3, 4}
        ):
            raise OfficialModelInputError(
                "period must identify regulation quarter one through four"
            )

        (
            receive_second_half_kickoff,
            home,
            half_seconds_remaining,
            game_seconds_remaining,
            diff_time_ratio,
            score_differential,
            down,
            yards_to_go,
            yardline_100,
            posteam_timeouts,
            defteam_timeouts,
        ) = values
        expected_home = float(self.posteam == self.home_team)
        if home != expected_home:
            raise OfficialModelInputError(
                "home orientation does not match possession team"
            )
        if receive_second_half_kickoff not in {0.0, 1.0}:
            raise OfficialModelInputError(
                "receive_2h_ko must be a binary indicator"
            )
        if not 0.0 <= game_seconds_remaining <= 3600.0:
            raise OfficialModelInputError(
                "game_seconds_remaining is outside regulation"
            )
        quarter_clock_bounds = {
            1: (2700.0, 3600.0),
            2: (1800.0, 2700.0),
            3: (900.0, 1800.0),
            4: (0.0, 900.0),
        }
        lower_clock, upper_clock = quarter_clock_bounds[self.period]
        if not lower_clock <= game_seconds_remaining <= upper_clock:
            raise OfficialModelInputError(
                "game clock is inconsistent with the regulation period"
            )
        expected_half_seconds = (
            game_seconds_remaining - 1800.0
            if self.period <= 2
            else game_seconds_remaining
        )
        if half_seconds_remaining != expected_half_seconds:
            raise OfficialModelInputError(
                "clock fields are not a coherent regulation state"
            )
        if (
            self.period > 2
            and receive_second_half_kickoff != 0.0
        ):
            raise OfficialModelInputError(
                "receive_2h_ko cannot remain set after halftime"
            )
        elapsed_share = (3600.0 - game_seconds_remaining) / 3600.0
        expected_ratio = score_differential / math.exp(-4.0 * elapsed_share)
        if not math.isclose(
            diff_time_ratio,
            expected_ratio,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise OfficialModelInputError(
                "Diff_Time_Ratio does not match score and game clock"
            )
        if down not in {1.0, 2.0, 3.0, 4.0}:
            raise OfficialModelInputError("down must be one through four")
        if yards_to_go < 0.0:
            raise OfficialModelInputError("ydstogo must be nonnegative")
        if not 0.0 <= yardline_100 <= 100.0:
            raise OfficialModelInputError(
                "yardline_100 must be between zero and one hundred"
            )
        if (
            posteam_timeouts not in {0.0, 1.0, 2.0, 3.0}
            or defteam_timeouts not in {0.0, 1.0, 2.0, 3.0}
        ):
            raise OfficialModelInputError(
                "regulation timeouts must be integers from zero through three"
            )


def _require_runtime_version() -> None:
    if xgboost.__version__ != REQUIRED_XGBOOST_VERSION:
        raise OfficialModelAssetError(
            "xgboost runtime version does not match the frozen reproduction"
        )


def _require_registry_binding(program_root: Path) -> None:
    try:
        matches = [
            row
            for row in load_dataset_registry(program_root)
            if row.dataset_id == DATASET_ID
        ]
    except ResearchRegistryError as error:
        raise OfficialModelAssetError(
            "dataset registry cannot prove the official model asset"
        ) from error
    if len(matches) != 1:
        raise OfficialModelAssetError(
            "dataset registry must contain exactly one official model asset"
        )
    row = matches[0]
    if (
        row.catalog_item_ids != ("I-018",)
        or row.canonical_url != ASSET_URL
        or row.license != "MIT"
        or row.license_status != "approved"
        or row.allowed_experiments != ("X-11",)
        or row.manifest_sha256 != ASSET_MANIFEST_SHA256
        or row.status != "registered"
    ):
        raise OfficialModelAssetError(
            "dataset registry differs from the frozen official asset binding"
        )


def _require_verified_asset_identity(verified: VerifiedStaticObject) -> None:
    record = verified.record
    manifest = record.manifest
    actual_request = dict(manifest.source_request)
    if (
        record.source != "nflverse"
        or record.dataset != DATASET_ID
        or record.version != ARCHIVE_TAG_COMMIT
        or record.partition != f"asset-{ASSET_ID}"
        or record.extension != "ubj"
        or manifest.manifest_sha256 != ASSET_MANIFEST_SHA256
        or manifest.dataset_id != DATASET_ID
        or manifest.object_kind != "byte_exact_original"
        or manifest.source_url != ASSET_URL
        or actual_request != dict(EXPECTED_SOURCE_REQUEST)
        or manifest.source_cursor != EXPECTED_SOURCE_CURSOR
        or manifest.byte_length != ASSET_BYTE_LENGTH
        or manifest.object_sha256 != ASSET_SHA256
        or manifest.media_type != "application/octet-stream"
        or manifest.schema_fingerprint != ASSET_SCHEMA_FINGERPRINT
        or manifest.license_ref != "I-018"
        or manifest.license_status != "approved"
        or manifest.upstream_partition != f"asset-{ASSET_ID}"
        or len(verified.object_bytes) != ASSET_BYTE_LENGTH
    ):
        raise OfficialModelAssetError(
            "verified bytes or manifest differ from the frozen asset identity"
        )


class OfficialNoSpreadPredictor:
    """A preloaded predictor with no public model mutation surface."""

    __slots__ = ("__booster",)

    def __init__(self, booster: xgboost.Booster) -> None:
        object.__setattr__(self, "_OfficialNoSpreadPredictor__booster", booster)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("official predictor is immutable")

    def _predict_array(
        self,
        inputs: tuple[NoSpreadModelInput, ...],
    ) -> tuple[float, ...]:
        if not inputs:
            raise OfficialModelInputError(
                "prediction batch must contain at least one input"
            )
        if any(not isinstance(item, NoSpreadModelInput) for item in inputs):
            raise OfficialModelInputError(
                "predictor accepts only validated no-spread model inputs"
            )
        matrix = np.asarray(
            [item.feature_values for item in inputs],
            dtype=np.float32,
        )
        probabilities = self.__booster.inplace_predict(matrix)
        result = tuple(float(value) for value in probabilities)
        if (
            len(result) != len(inputs)
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in result
            )
        ):
            raise OfficialModelAssetError(
                "official booster returned invalid probabilities"
            )
        return result

    def predict_possession(self, model_input: NoSpreadModelInput) -> float:
        """Return the official possession-team win probability."""

        return self._predict_array((model_input,))[0]

    def predict_possession_batch(
        self,
        inputs: Iterable[NoSpreadModelInput],
    ) -> tuple[float, ...]:
        """Return possession-team probabilities without reloading the booster."""

        return self._predict_array(tuple(inputs))

    def predict_home(self, model_input: NoSpreadModelInput) -> float:
        """Orient the official possession-team output to the home team."""

        possession_probability = self.predict_possession(model_input)
        if model_input.posteam == model_input.home_team:
            return possession_probability
        return 1.0 - possession_probability


def load_official_no_spread_predictor(
    *,
    program_root: str | Path,
    store_root: str | Path,
    manifest_path: str | Path,
) -> OfficialNoSpreadPredictor:
    """Verify the immutable asset and preload the exact official booster."""

    program = Path(program_root).resolve()
    _require_runtime_version()
    _require_registry_binding(program)
    verified = read_verified_static_object(
        manifest_path,
        store_root=store_root,
        program_root=program,
    )
    _require_verified_asset_identity(verified)

    booster = xgboost.Booster()
    try:
        booster.load_model(bytearray(verified.object_bytes))
    except xgboost.core.XGBoostError as error:
        raise OfficialModelAssetError(
            "verified official asset cannot be loaded as XGBoost UBJSON"
        ) from error
    if booster.num_features() != EXPECTED_FEATURE_COUNT:
        raise OfficialModelAssetError(
            "official booster feature count differs from the frozen contract"
        )
    if booster.num_boosted_rounds() != EXPECTED_BOOSTED_ROUNDS:
        raise OfficialModelAssetError(
            "official booster boosted round count differs from the frozen contract"
        )
    return OfficialNoSpreadPredictor(booster)


__all__ = [
    "ARCHIVE_TAG_COMMIT",
    "ASSET_BYTE_LENGTH",
    "ASSET_ID",
    "ASSET_MANIFEST_SHA256",
    "ASSET_SCHEMA_FINGERPRINT",
    "ASSET_SHA256",
    "ASSET_URL",
    "EXPECTED_BOOSTED_ROUNDS",
    "EXPECTED_FEATURE_COUNT",
    "EXPECTED_SOURCE_CURSOR",
    "EXPECTED_SOURCE_REQUEST",
    "FEATURE_HELPER_COMMIT",
    "FEATURE_NAMES",
    "FEATURE_SPEC_COMMIT",
    "GOLDEN_VECTOR_EVIDENCE_CLASS",
    "NoSpreadModelInput",
    "OfficialModelAssetError",
    "OfficialModelInputError",
    "OfficialNoSpreadPredictor",
    "REQUIRED_XGBOOST_VERSION",
    "load_official_no_spread_predictor",
]
