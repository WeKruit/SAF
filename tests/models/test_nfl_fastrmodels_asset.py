from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

from prediction_market.models import nfl_fastrmodels  # noqa: E402
from prediction_market.static_store import StaticStoreError  # noqa: E402


EXPECTED_FEATURES = (
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
GOLDEN_VALUES = (
    1.0,
    1.0,
    600.0,
    2400.0,
    26.555675506591797,
    7.0,
    2.0,
    4.0,
    45.0,
    2.0,
    1.0,
)


def _official_manifest(program_root: Path = PROJECT_ROOT) -> Path:
    store_root = program_root / "var" / "raw"
    matches = list(
        (
            store_root
            / "manifests"
            / "source=nflverse"
            / "dataset=DS-NFL-FASTRMODELS"
        ).rglob("*.manifest.json")
    )
    if len(matches) != 1:
        pytest.skip("official no-spread model manifest is not present uniquely")
    manifest_path = matches[0]
    document = json.loads(manifest_path.read_bytes())
    native_object_path = document["native_object_path"]
    if type(native_object_path) is not str:
        pytest.fail("official manifest native_object_path must be a string")
    relative_object_path = Path(native_object_path)
    if (
        relative_object_path.is_absolute()
        or ".." in relative_object_path.parts
    ):
        pytest.fail("official manifest native_object_path must stay in raw root")
    object_path = store_root / relative_object_path
    if object_path.is_symlink() or not object_path.is_file():
        pytest.skip(
            "official no-spread raw object is not present as a "
            "non-symlink regular file"
        )
    try:
        object_path.resolve(strict=True).relative_to(
            store_root.resolve(strict=True)
        )
    except (OSError, ValueError):
        pytest.skip("official no-spread raw object is outside the raw root")
    return manifest_path


def test_official_manifest_skips_when_tracked_manifest_has_no_raw_object(
    tmp_path: Path,
) -> None:
    source_manifests = list(
        (
            PROJECT_ROOT
            / "var"
            / "raw"
            / "manifests"
            / "source=nflverse"
            / "dataset=DS-NFL-FASTRMODELS"
        ).rglob("*.manifest.json")
    )
    assert len(source_manifests) == 1
    source_manifest = source_manifests[0]
    program_root = tmp_path / "clean-clone"
    copied_manifest = program_root / source_manifest.relative_to(PROJECT_ROOT)
    copied_manifest.parent.mkdir(parents=True)
    shutil.copy2(source_manifest, copied_manifest)

    with pytest.raises(pytest.skip.Exception, match="raw object"):
        _official_manifest(program_root)


def _copy_governance_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "program"
    (root / "charter").mkdir(parents=True)
    (root / "registries").mkdir()
    shutil.copy2(
        PROJECT_ROOT / "charter" / "catalog_registry.csv",
        root / "charter" / "catalog_registry.csv",
    )
    for filename in (
        "data_license_register.csv",
        "dataset_registry.csv",
        "experiment_registry.csv",
    ):
        shutil.copy2(
            PROJECT_ROOT / "registries" / filename,
            root / "registries" / filename,
        )
    return root


def _replace_license_review(
    root: Path,
    catalog_item_id: str,
    **changes: str,
) -> None:
    path = root / "registries" / "data_license_register.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, strict=True))
    matches = [row for row in rows if row["catalog_item_id"] == catalog_item_id]
    assert len(matches) == 1
    matches[0].update(changes)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _input(
    *,
    names: tuple[str, ...] = EXPECTED_FEATURES,
    values: tuple[float, ...] = GOLDEN_VALUES,
    posteam: str = "HME",
    home_team: str = "HME",
    away_team: str = "AWY",
    period: int = 2,
) -> nfl_fastrmodels.NoSpreadModelInput:
    return nfl_fastrmodels.NoSpreadModelInput(
        feature_names=names,
        feature_values=values,
        posteam=posteam,
        home_team=home_team,
        away_team=away_team,
        period=period,
    )


def _verified_asset_fixture(
    *,
    byte_length: int = 106951,
    object_sha256: str = (
        "sha256:"
        "ff58c98d22e3e36f50f6e82f8e68d3ed0920101132a50d30bb70291df33c0a4c"
    ),
) -> SimpleNamespace:
    return SimpleNamespace(
        record=SimpleNamespace(
            source="nflverse",
            dataset="DS-NFL-FASTRMODELS",
            version="9f2495fdb4943087ca663d96706eb5df7973aff4",
            partition="asset-253928623",
            extension="ubj",
            manifest=SimpleNamespace(
                manifest_sha256=nfl_fastrmodels.ASSET_MANIFEST_SHA256,
                dataset_id="DS-NFL-FASTRMODELS",
                object_kind="byte_exact_original",
                source_url=nfl_fastrmodels.ASSET_URL,
                source_request=nfl_fastrmodels.EXPECTED_SOURCE_REQUEST,
                source_cursor=nfl_fastrmodels.EXPECTED_SOURCE_CURSOR,
                byte_length=byte_length,
                object_sha256=object_sha256,
                media_type="application/octet-stream",
                schema_fingerprint=nfl_fastrmodels.ASSET_SCHEMA_FINGERPRINT,
                license_ref="I-018",
                license_status="approved",
                upstream_partition="asset-253928623",
            ),
        ),
        object_bytes=b"\x00" * byte_length,
    )


def test_frozen_official_identity_and_feature_order() -> None:
    assert nfl_fastrmodels.MODEL_ID == (
        "MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1"
    )
    assert nfl_fastrmodels.ASSET_ID == 253928623
    assert nfl_fastrmodels.ASSET_BYTE_LENGTH == 106951
    assert nfl_fastrmodels.ASSET_SHA256 == (
        "sha256:"
        "ff58c98d22e3e36f50f6e82f8e68d3ed0920101132a50d30bb70291df33c0a4c"
    )
    assert nfl_fastrmodels.ARCHIVE_TAG_COMMIT == (
        "9f2495fdb4943087ca663d96706eb5df7973aff4"
    )
    assert nfl_fastrmodels.FEATURE_SPEC_COMMIT == (
        "75c7b68bc49535370236c38c9826265da075bd71"
    )
    assert nfl_fastrmodels.FEATURE_HELPER_COMMIT == (
        "ead5e2f9641490f692d923c04835bd3b90275b4e"
    )
    assert nfl_fastrmodels.FEATURE_NAMES == EXPECTED_FEATURES
    assert nfl_fastrmodels.EXPECTED_FEATURE_COUNT == 11
    assert nfl_fastrmodels.EXPECTED_BOOSTED_ROUNDS == 65
    assert nfl_fastrmodels.REQUIRED_XGBOOST_VERSION == "3.3.0"


def test_missing_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(StaticStoreError):
        nfl_fastrmodels.load_official_no_spread_predictor(
            program_root=PROJECT_ROOT,
            store_root=tmp_path,
            manifest_path=tmp_path / "missing.manifest.json",
        )


@pytest.mark.parametrize(
    ("status", "commercial_use", "operational_use"),
    [
        ("NOT_GREEN_OPEN", "UNKNOWN", "RESEARCH_ONLY"),
        ("NOT_GREEN_BLOCKED", "PROHIBITED", "BLOCKED"),
    ],
)
def test_team_i_license_downgrade_fails_before_asset_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
    commercial_use: str,
    operational_use: str,
) -> None:
    root = _copy_governance_fixture(tmp_path)
    _replace_license_review(
        root,
        "I-018",
        status=status,
        commercial_use=commercial_use,
        redistribution="UNKNOWN",
        attribution_required="UNKNOWN",
        operational_use=operational_use,
        open_blocker="Team I approval withdrawn",
        approval_ref="",
    )
    asset_read = False

    def unexpected_asset_read(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal asset_read
        asset_read = True
        raise AssertionError("asset bytes must not be read after license downgrade")

    monkeypatch.setattr(
        nfl_fastrmodels,
        "read_verified_static_object",
        unexpected_asset_read,
    )

    with pytest.raises(
        nfl_fastrmodels.OfficialModelAssetError,
        match="dataset registry cannot prove",
    ):
        nfl_fastrmodels.load_official_no_spread_predictor(
            program_root=root,
            store_root=tmp_path,
            manifest_path=Path("unused"),
        )
    assert asset_read is False


def test_verified_reader_rejects_tampered_model_bytes(tmp_path: Path) -> None:
    source_manifest = _official_manifest()
    source_store = PROJECT_ROOT / "var" / "raw"
    document = json.loads(source_manifest.read_bytes())
    object_relative = Path(document["native_object_path"])
    copied_manifest = tmp_path / source_manifest.relative_to(source_store)
    copied_object = tmp_path / object_relative
    copied_manifest.parent.mkdir(parents=True)
    copied_object.parent.mkdir(parents=True)
    shutil.copy2(source_manifest, copied_manifest)
    shutil.copy2(source_store / object_relative, copied_object)
    payload = bytearray(copied_object.read_bytes())
    payload[-1] ^= 1
    copied_object.chmod(0o600)
    copied_object.write_bytes(payload)

    with pytest.raises(StaticStoreError, match="SHA-256"):
        nfl_fastrmodels.load_official_no_spread_predictor(
            program_root=PROJECT_ROOT,
            store_root=tmp_path,
            manifest_path=copied_manifest,
        )


def test_verified_reader_rejects_tampered_manifest(tmp_path: Path) -> None:
    source_manifest = _official_manifest()
    source_store = PROJECT_ROOT / "var" / "raw"
    document = json.loads(source_manifest.read_bytes())
    object_relative = Path(document["native_object_path"])
    copied_manifest = tmp_path / source_manifest.relative_to(source_store)
    copied_object = tmp_path / object_relative
    copied_manifest.parent.mkdir(parents=True)
    copied_object.parent.mkdir(parents=True)
    shutil.copy2(source_manifest, copied_manifest)
    shutil.copy2(source_store / object_relative, copied_object)
    document["coverage"] = "tampered"
    copied_manifest.chmod(0o600)
    copied_manifest.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StaticStoreError):
        nfl_fastrmodels.load_official_no_spread_predictor(
            program_root=PROJECT_ROOT,
            store_root=tmp_path,
            manifest_path=copied_manifest,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("github_release_asset_id", 253928624, "asset identity"),
        ("archive_tag_commit", "0" * 40, "asset identity"),
    ],
)
def test_wrong_source_identity_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    message: str,
) -> None:
    request = dict(nfl_fastrmodels.EXPECTED_SOURCE_REQUEST)
    request[field] = replacement
    verified = SimpleNamespace(
        record=SimpleNamespace(
            source="nflverse",
            dataset="DS-NFL-FASTRMODELS",
            version=nfl_fastrmodels.ARCHIVE_TAG_COMMIT,
            partition="asset-253928623",
            extension="ubj",
            manifest=SimpleNamespace(
                manifest_sha256=nfl_fastrmodels.ASSET_MANIFEST_SHA256,
                dataset_id="DS-NFL-FASTRMODELS",
                object_kind="byte_exact_original",
                source_url=nfl_fastrmodels.ASSET_URL,
                source_request=request,
                source_cursor=nfl_fastrmodels.EXPECTED_SOURCE_CURSOR,
                byte_length=nfl_fastrmodels.ASSET_BYTE_LENGTH,
                object_sha256=nfl_fastrmodels.ASSET_SHA256,
                media_type="application/octet-stream",
                schema_fingerprint=nfl_fastrmodels.ASSET_SCHEMA_FINGERPRINT,
                license_ref="I-018",
                license_status="approved",
                upstream_partition="asset-253928623",
            ),
        ),
        object_bytes=b"not loaded when identity is wrong",
    )
    monkeypatch.setattr(
        nfl_fastrmodels,
        "read_verified_static_object",
        lambda *args, **kwargs: verified,
    )

    with pytest.raises(
        nfl_fastrmodels.OfficialModelAssetError,
        match=message,
    ):
        nfl_fastrmodels.load_official_no_spread_predictor(
            program_root=PROJECT_ROOT,
            store_root=PROJECT_ROOT / "var" / "raw",
            manifest_path=Path("unused"),
        )


@pytest.mark.parametrize(
    ("byte_length", "sha256"),
    [
        (106950, nfl_fastrmodels.ASSET_SHA256),
        (106951, "sha256:" + "0" * 64),
    ],
)
def test_wrong_length_or_sha_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    byte_length: int,
    sha256: str,
) -> None:
    verified = _verified_asset_fixture(
        byte_length=byte_length,
        object_sha256=sha256,
    )
    monkeypatch.setattr(
        nfl_fastrmodels,
        "read_verified_static_object",
        lambda *args, **kwargs: verified,
    )

    with pytest.raises(
        nfl_fastrmodels.OfficialModelAssetError,
        match="asset identity",
    ):
        nfl_fastrmodels.load_official_no_spread_predictor(
            program_root=PROJECT_ROOT,
            store_root=PROJECT_ROOT / "var" / "raw",
            manifest_path=Path("unused"),
        )


def test_wrong_runtime_version_fails_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nfl_fastrmodels.xgboost, "__version__", "3.2.1")

    with pytest.raises(
        nfl_fastrmodels.OfficialModelAssetError,
        match="xgboost runtime",
    ):
        nfl_fastrmodels.load_official_no_spread_predictor(
            program_root=PROJECT_ROOT,
            store_root=PROJECT_ROOT / "var" / "raw",
            manifest_path=Path("unused"),
        )


@pytest.mark.parametrize(
    ("feature_count", "round_count", "message"),
    [
        (10, 65, "feature count"),
        (11, 64, "boosted round count"),
    ],
)
def test_wrong_booster_shape_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    feature_count: int,
    round_count: int,
    message: str,
) -> None:
    class FakeBooster:
        def load_model(self, value: bytearray) -> None:
            assert isinstance(value, bytearray)

        def num_features(self) -> int:
            return feature_count

        def num_boosted_rounds(self) -> int:
            return round_count

    monkeypatch.setattr(
        nfl_fastrmodels,
        "read_verified_static_object",
        lambda *args, **kwargs: _verified_asset_fixture(),
    )
    monkeypatch.setattr(nfl_fastrmodels.xgboost, "Booster", FakeBooster)

    with pytest.raises(
        nfl_fastrmodels.OfficialModelAssetError,
        match=message,
    ):
        nfl_fastrmodels.load_official_no_spread_predictor(
            program_root=PROJECT_ROOT,
            store_root=PROJECT_ROOT / "var" / "raw",
            manifest_path=Path("unused"),
        )


def test_input_rejects_wrong_feature_order() -> None:
    names = list(EXPECTED_FEATURES)
    names[0], names[1] = names[1], names[0]
    with pytest.raises(
        nfl_fastrmodels.OfficialModelInputError,
        match="feature names or order",
    ):
        _input(names=tuple(names))


@pytest.mark.parametrize("replacement", [float("nan"), float("inf")])
def test_input_rejects_nonfinite_values(replacement: float) -> None:
    values = list(GOLDEN_VALUES)
    values[4] = replacement
    with pytest.raises(
        nfl_fastrmodels.OfficialModelInputError,
        match="finite",
    ):
        _input(values=tuple(values))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"posteam": "UNK"}, "game participant"),
        ({"values": (*GOLDEN_VALUES[:1], 0.0, *GOLDEN_VALUES[2:])}, "home orientation"),
        (
            {
                "values": (
                    *GOLDEN_VALUES[:2],
                    599.0,
                    *GOLDEN_VALUES[3:],
                )
            },
            "clock fields",
        ),
    ],
)
def test_input_rejects_ineligible_or_orientation_mutated_state(
    changes: dict[str, object],
    message: str,
) -> None:
    kwargs = {
        "posteam": "HME",
        "values": GOLDEN_VALUES,
    }
    kwargs.update(changes)
    with pytest.raises(
        nfl_fastrmodels.OfficialModelInputError,
        match=message,
    ):
        _input(
            posteam=str(kwargs["posteam"]),
            values=tuple(kwargs["values"]),  # type: ignore[arg-type]
        )


def test_period_disambiguates_halftime_clock_boundary() -> None:
    first_half_values = list(GOLDEN_VALUES)
    first_half_values[2] = 0.0
    first_half_values[3] = 1800.0
    first_half_values[4] = 7.0 / math.exp(-2.0)
    first_half = _input(
        values=tuple(first_half_values),
        period=2,
    )
    assert first_half.feature_values[0] == 1.0

    second_half_values = list(first_half_values)
    second_half_values[0] = 0.0
    second_half_values[2] = 1800.0
    second_half = _input(
        values=tuple(second_half_values),
        period=3,
    )
    assert second_half.feature_values[0] == 0.0

    for period, wrong_half_seconds in ((2, 1800.0), (3, 0.0)):
        incoherent = list(
            first_half_values if period == 2 else second_half_values
        )
        incoherent[2] = wrong_half_seconds
        with pytest.raises(
            nfl_fastrmodels.OfficialModelInputError,
            match="clock fields",
        ):
            _input(values=tuple(incoherent), period=period)


@pytest.mark.parametrize(
    ("values", "posteam", "home_team", "away_team"),
    [
        (
            (
                1.0,
                0.0,
                0.0,
                1800.0,
                29.556224395722598,
                4.0,
                1.0,
                5.0,
                19.0,
                0.0,
                3.0,
            ),
            "IND",
            "BAL",
            "IND",
        ),
        (
            (
                0.0,
                0.0,
                0.0,
                1800.0,
                -96.05772928609845,
                -13.0,
                1.0,
                10.0,
                12.0,
                1.0,
                0.0,
            ),
            "TB",
            "WAS",
            "TB",
        ),
    ],
)
def test_frozen_2021_halftime_rows_are_valid_official_inputs(
    values: tuple[float, ...],
    posteam: str,
    home_team: str,
    away_team: str,
) -> None:
    model_input = _input(
        values=values,
        posteam=posteam,
        home_team=home_team,
        away_team=away_team,
        period=2,
    )

    assert model_input.feature_values == values


def test_official_asset_golden_vector_and_home_away_orientation() -> None:
    predictor = nfl_fastrmodels.load_official_no_spread_predictor(
        program_root=PROJECT_ROOT,
        store_root=PROJECT_ROOT / "var" / "raw",
        manifest_path=_official_manifest(),
    )
    home_input = _input()
    away_values = list(GOLDEN_VALUES)
    away_values[1] = 0.0
    away_input = _input(
        values=tuple(away_values),
        posteam="AWY",
    )

    possession_probability = predictor.predict_possession(home_input)
    assert possession_probability == pytest.approx(
        0.84001481533050537,
        abs=1e-12,
    )
    assert predictor.predict_home(home_input) == pytest.approx(
        possession_probability,
        abs=0,
    )
    away_possession_probability = predictor.predict_possession(away_input)
    assert predictor.predict_home(away_input) == pytest.approx(
        1.0 - away_possession_probability,
        abs=1e-12,
    )
    assert predictor.predict_possession_batch(
        (home_input, away_input)
    ) == pytest.approx(
        (possession_probability, away_possession_probability),
        abs=1e-12,
    )


def test_preloaded_predictor_has_no_public_mutation_surface() -> None:
    predictor = nfl_fastrmodels.load_official_no_spread_predictor(
        program_root=PROJECT_ROOT,
        store_root=PROJECT_ROOT / "var" / "raw",
        manifest_path=_official_manifest(),
    )

    with pytest.raises(AttributeError, match="immutable"):
        predictor.model = object()  # type: ignore[attr-defined]
    assert not hasattr(predictor, "booster")


def test_golden_vector_is_explicitly_engineering_smoke_only() -> None:
    assert nfl_fastrmodels.GOLDEN_VECTOR_EVIDENCE_CLASS == (
        "runtime_regression_fixture_not_accuracy_oracle"
    )
