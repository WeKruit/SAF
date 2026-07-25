from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, is_dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from prediction_market.sports import nfl_x13_spec as spec

EXPECTED_BINDINGS = {
    "2025_01_BAL_BUF": (
        "Core",
        "2025-09-07",
        "BAL",
        "BUF",
        40,
        41,
        ("2PT", "HIGH"),
        "40828",
        "584347",
        "0x7f2e645fd40487882cc9e7a05d240f5b1c2220a361fbb345b21f921070411409",
        "KXNFLGAME-25SEP07BALBUF",
    ),
    "2025_03_ARI_SF": (
        "Core",
        "2025-09-21",
        "ARI",
        "SF",
        15,
        16,
        ("SAF", "LOW"),
        "44348",
        "597273",
        "0x86114dd0c8cb0620543ac867b2c1c38715f6c71e62d2f6580c51c5223e6f3c3e",
        "KXNFLGAME-25SEP21ARISF",
    ),
    "2025_03_CIN_MIN": (
        "Core",
        "2025-09-21",
        "CIN",
        "MIN",
        10,
        48,
        ("RTD2", "BLOW"),
        "44339",
        "597264",
        "0x00cfb1fe371a24b3a9d81f1bb9ba5a037a53f70cf5f2741f3e8d24ca0eb46591",
        "KXNFLGAME-25SEP21CINMIN",
    ),
    "2025_03_LA_PHI": (
        "Rare",
        "2025-09-21",
        "LA",
        "PHI",
        26,
        33,
        ("BLK2",),
        "44341",
        "597266",
        "0xbaef857e53f69249525c725e35c4dafb4c9df13c915738351bf2d5ab9fef01b9",
        "KXNFLGAME-25SEP21LAPHI",
    ),
    "2025_03_NYJ_TB": (
        "Core",
        "2025-09-21",
        "NYJ",
        "TB",
        27,
        29,
        ("RTD", "BLK"),
        "44342",
        "597267",
        "0x44d172447f68fcc304a574af3439776f798155fee1ea6ade6861ea62d269a856",
        "KXNFLGAME-25SEP21NYJTB",
    ),
    "2025_04_GB_DAL": (
        "Rare",
        "2025-09-28",
        "GB",
        "DAL",
        40,
        40,
        ("def2PT", "OT", "TIE"),
        "47514",
        "606182",
        "0x0af48a189e649217f4c5af8184a22e3961500c63e97e9474d010a93350e0b4c7",
        "KXNFLGAME-25SEP28GBDAL",
    ),
    "2025_04_PHI_TB": (
        "Rare",
        "2025-09-28",
        "PHI",
        "TB",
        31,
        25,
        ("SAF", "BLK"),
        "47509",
        "606177",
        "0xccc244fc8e622c7a3768672661e64b7e9bb53bbb4f1e53a3aa36bd0d33428548",
        "KXNFLGAME-25SEP28PHITB",
    ),
    "2025_08_CLE_NE": (
        "Rare",
        "2025-10-26",
        "CLE",
        "NE",
        13,
        32,
        ("SAF", "ONS", "2PT"),
        "61641",
        "639540",
        "0xcfcbaa9a402806aa70910ddc64179d29af5bdbac676ec79ac600912f0db26c1b",
        "KXNFLGAME-25OCT26CLENE",
    ),
    "2025_08_NYJ_CIN": (
        "Core",
        "2025-10-26",
        "NYJ",
        "CIN",
        39,
        38,
        ("HIGH",),
        "61639",
        "639538",
        "0x983948ed0de118d2567507ce97f3275ffd04edb1b94a427e7520e6b1161a23a1",
        "KXNFLGAME-25OCT26NYJCIN",
    ),
    "2025_09_CHI_CIN": (
        "Rare",
        "2025-11-02",
        "CHI",
        "CIN",
        47,
        42,
        ("RTD", "2PT", "BLK", "ONS"),
        "65485",
        "650203",
        "0xb6059f90055603d5d41cddec29021d91614ac5398cb1a0aa4d6c641189578744",
        "KXNFLGAME-25NOV02CHICIN",
    ),
    "2025_10_JAX_HOU": (
        "Rare",
        "2025-11-09",
        "JAX",
        "HOU",
        29,
        36,
        ("RTD2", "2PT", "NUL"),
        "70850",
        "660423",
        "0xf1afc8d6d150bb942af398a2e615ddfadf83517234e32ba67adc47feda90d803",
        "KXNFLGAME-25NOV09JACHOU",
    ),
    "2025_13_NO_MIA": (
        "Rare",
        "2025-11-30",
        "NO",
        "MIA",
        17,
        21,
        ("def2PT", "ONS"),
        "89356",
        "700733",
        "0x04f68fed65c7436d84cd952dd334c5fe543e54a32b0d7f148d494e1eb8301a13",
        "KXNFLGAME-25NOV30NOMIA",
    ),
    "2025_14_DAL_DET": (
        "Core",
        "2025-12-04",
        "DAL",
        "DET",
        30,
        44,
        ("HIGH", "BLK"),
        "89365",
        "700742",
        "0xe6f492eb15583ae324d43e987e3ad1b7bd1a86690b3b62a2843162278ff7af0e",
        "KXNFLGAME-25DEC04DALDET",
    ),
    "2025_14_PHI_LAC": (
        "Core",
        "2025-12-08",
        "PHI",
        "LAC",
        19,
        22,
        ("OT", "MISS", "NUL"),
        "90035",
        "703065",
        "0x973a05cb855049840adb639aa0c927d4d0f5ee7cc383269a24256b39ec4cd2d9",
        "KXNFLGAME-25DEC08PHILAC",
    ),
    "2025_14_SEA_ATL": (
        "Core",
        "2025-12-07",
        "SEA",
        "ATL",
        37,
        9,
        ("RTD", "BLK", "BLOW"),
        "89366",
        "700743",
        "0x07fdb434a81996e35e05aeb4d7e8d9f79866a6ed435282489b50c9a781c0734a",
        "KXNFLGAME-25DEC07SEAATL",
    ),
    "2025_16_LA_SEA": (
        "Core",
        "2025-12-18",
        "LA",
        "SEA",
        37,
        38,
        ("OT", "RTD", "2PT"),
        "97878",
        "832734",
        "0x6749b632cb9df8883a4958aa5493e50f9bc71dd6597dcefaeb7ee401c2c790ae",
        "KXNFLGAME-25DEC18LASEA",
    ),
    "2025_18_KC_LV": (
        "Rare",
        "2026-01-04",
        "KC",
        "LV",
        12,
        14,
        ("SAF", "LOW", "noTD"),
        "115293",
        "988312",
        "0x7b718a0402aa0bb771e441d60402fef788a965c3a4b79d2de5fc053dc705cf45",
        "KXNFLGAME-26JAN04KCLV",
    ),
    "2025_19_LA_CAR": (
        "Core",
        "2026-01-10",
        "LA",
        "CAR",
        34,
        31,
        ("POST", "BLK"),
        "145825",
        "1116464",
        "0x32102e4ab8969621cab85949c3c571b5008a4aa30d31022fa659d0ed287686a9",
        "KXNFLGAME-26JAN10LACAR",
    ),
    "2025_20_BUF_DEN": (
        "Core",
        "2026-01-17",
        "BUF",
        "DEN",
        30,
        33,
        ("POST", "OT"),
        "161553",
        "1174855",
        "0x746f66fb1859ce9f795b0bbe0b4fede7666bfe4de95b2aa280296b25f763deee",
        "KXNFLGAME-26JAN17BUFDEN",
    ),
    "2025_20_HOU_NE": (
        "Core",
        "2026-01-18",
        "HOU",
        "NE",
        16,
        28,
        ("POST", "RTD"),
        "161554",
        "1174856",
        "0x4195016305dc0e1b849be8f3a39721c57c9a2c9717b9a050abfd10014de03078",
        "KXNFLGAME-26JAN18HOUNE",
    ),
}


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _game_artifact_sha256s() -> tuple[tuple[str, str], ...]:
    return tuple(
        (game_id, _digest("c")) for game_id in spec.X13_GAME_IDS
    )


def _primitive_contract(**changes: object) -> spec.MarketContractV1:
    values: dict[str, object] = {
        "contract_id": "kalshi:KXNFLGAME-25DEC04DALDET-DAL",
        "logical_market_id": "kalshi:KXNFLGAME-25DEC04DALDET:winner",
        "venue": "kalshi",
        "family": "moneyline",
        "period": "full_game",
        "subject": "DAL",
        "measure": "wins",
        "comparator": "eq",
        "line": None,
        "outcomes": ("yes", "no"),
        "rule_sha256": _digest("a"),
        "dependency_game_ids": ("2025_14_DAL_DET",),
        "kind": "primitive",
        "analysis_eligible": True,
    }
    values.update(changes)
    return spec.MarketContractV1(**values)


def _observation(**changes: object) -> spec.MarketObservationV1:
    values: dict[str, object] = {
        "observation_id": "obs-1",
        "contract_id": "contract-1",
        "venue": "kalshi",
        "observation_kind": "trade",
        "source_interval_start": datetime(2025, 12, 5, 1, tzinfo=UTC),
        "source_interval_end": datetime(
            2025, 12, 5, 1, 0, 1, tzinfo=UTC
        ),
        "price": Decimal("0.51"),
        "size": Decimal(10),
        "outcome": "yes",
        "observed": True,
        "bid": Decimal("0.50"),
        "ask": Decimal("0.52"),
    }
    values.update(changes)
    return spec.MarketObservationV1(**values)


def test_frozen_batch_spec_has_exact_sorted_twenty_game_design() -> None:
    batch = spec.X13_BATCH_SPEC

    assert batch.experiment_id == "X-13"
    assert batch.status == "PRELIMINARY_SOURCE_TIME_ONLY"
    assert batch.game_ids == tuple(sorted(EXPECTED_BINDINGS))
    assert len(batch.game_ids) == 20
    assert batch.horizons_seconds == (1, 2, 5, 10, 30, 60)
    assert batch.delay_scenarios_seconds == (0, 1, 2, 3, 5, 10)
    assert isinstance(batch.plan_id, str)
    assert batch.plan_id.startswith("sha256:")
    with pytest.raises(FrozenInstanceError):
        batch.season = 2026  # type: ignore[misc]


def test_all_twenty_native_games_have_exact_dual_venue_bindings() -> None:
    assert tuple(binding.native_game_id for binding in spec.X13_GAME_BINDINGS) == (
        tuple(sorted(EXPECTED_BINDINGS))
    )
    assert len(spec.X13_GAME_BINDINGS) == 20

    for binding in spec.X13_GAME_BINDINGS:
        expected = EXPECTED_BINDINGS[binding.native_game_id]
        actual = (
            binding.stratum,
            binding.game_date.isoformat(),
            binding.away_team,
            binding.home_team,
            binding.away_score,
            binding.home_score,
            binding.coverage_labels,
            binding.polymarket_event_id,
            binding.polymarket_market_id,
            binding.polymarket_condition_id,
            binding.kalshi_event_ticker,
        )
        assert actual == expected
        assert binding.polymarket_event_id
        assert binding.polymarket_market_id
        assert binding.polymarket_condition_id.startswith("0x")
        assert binding.kalshi_away_ticker == (
            f"{binding.kalshi_event_ticker}-"
            f"{spec.kalshi_team_code(binding.away_team)}"
        )
        assert binding.kalshi_home_ticker == (
            f"{binding.kalshi_event_ticker}-"
            f"{spec.kalshi_team_code(binding.home_team)}"
        )
        assert binding.scheduled_at_utc is None


def test_jax_aliases_to_jac_while_la_semantics_are_preserved() -> None:
    by_id = {
        binding.native_game_id: binding
        for binding in spec.X13_GAME_BINDINGS
    }

    assert spec.KALSHI_TEAM_ALIASES == (("JAX", "JAC"),)
    assert spec.kalshi_team_code("JAX") == "JAC"
    assert by_id["2025_10_JAX_HOU"].kalshi_away_ticker.endswith("-JAC")
    assert by_id["2025_10_JAX_HOU"].away_team == "JAX"
    assert spec.kalshi_team_code("LA") == "LA"
    assert by_id["2025_16_LA_SEA"].kalshi_away_ticker.endswith("-LA")
    assert by_id["2025_19_LA_CAR"].away_team == "LA"


def test_batch_spec_fails_closed_on_count_duplicates_and_wrong_set() -> None:
    batch = spec.X13_BATCH_SPEC

    with pytest.raises(ValueError, match="exactly 20"):
        replace(batch, game_ids=batch.game_ids[:-1])
    duplicate = tuple(sorted((*batch.game_ids[:-1], batch.game_ids[0])))
    with pytest.raises(ValueError, match="duplicate"):
        replace(batch, game_ids=duplicate)
    wrong = tuple(sorted((*batch.game_ids[:-1], "2025_01_ARI_NO")))
    with pytest.raises(ValueError, match="frozen X-13"):
        replace(batch, game_ids=wrong)


def test_native_binding_rejects_unknown_stratum_team_and_ticker_mismatch() -> None:
    binding = spec.X13_GAME_BINDINGS[0]
    jax = next(
        value
        for value in spec.X13_GAME_BINDINGS
        if value.native_game_id == "2025_10_JAX_HOU"
    )

    with pytest.raises(ValueError, match="stratum"):
        replace(binding, stratum="Other")
    with pytest.raises(ValueError, match="native_game_id"):
        replace(binding, away_team="BUF")
    with pytest.raises(ValueError, match="JAC"):
        replace(jax, kalshi_away_ticker=f"{jax.kalshi_event_ticker}-JAX")
    with pytest.raises(ValueError, match="event ticker"):
        replace(binding, kalshi_event_ticker="KXNFLGAME-25SEP08BALBUF")


def test_every_non_null_datetime_must_be_utc() -> None:
    binding = spec.X13_GAME_BINDINGS[0]
    non_utc = datetime(
        2025, 9, 7, 12, tzinfo=timezone(timedelta(hours=-5))
    )

    with pytest.raises(ValueError, match="UTC"):
        replace(binding, scheduled_at_utc=non_utc)
    with pytest.raises(ValueError, match="UTC"):
        _observation(source_interval_start=non_utc)
    with pytest.raises(ValueError, match="UTC"):
        spec.BatchManifestV1(
            plan_id=spec.X13_BATCH_SPEC.plan_id,
            raw_manifest_sha256s=(_digest("b"),),
            builder_versions=(("builder", "v1"),),
            game_artifact_sha256s=_game_artifact_sha256s(),
            published_at_utc=non_utc,
        )


def test_family_catalog_covers_primitives_and_separates_composites() -> None:
    catalog = spec.X13_VENUE_FAMILY_CATALOG
    expected_primitives = {
        "moneyline",
        "spread",
        "total",
        "period",
        "team_total",
        "player_passing",
        "player_rushing",
        "player_receiving",
        "player_receptions",
        "player_td",
        "first_td",
        "any_td",
    }

    assert expected_primitives <= set(catalog.primitive_families)
    assert catalog.same_game_composite_family == "same_game_composite"
    assert catalog.cross_game_inventory_family == "cross_game_mve_inventory"
    assert catalog.same_game_composite_family != (
        catalog.cross_game_inventory_family
    )
    assert set(catalog.all_families) == expected_primitives | {
        "same_game_composite",
        "cross_game_mve_inventory",
    }


def test_contracts_fail_closed_on_unknown_family_or_dependency() -> None:
    contract = _primitive_contract()
    assert contract.logical_market_id == (
        "kalshi:KXNFLGAME-25DEC04DALDET:winner"
    )
    with pytest.raises(ValueError, match="venue-namespaced"):
        _primitive_contract(contract_id="KXNFLGAME-25DEC04DALDET-DAL")
    with pytest.raises(ValueError, match="venue-namespaced"):
        _primitive_contract(
            venue="polymarket",
            contract_id="kalshi:KXNFLGAME-25DEC04DALDET-DAL",
        )
    with pytest.raises(ValueError, match="unknown contract family"):
        _primitive_contract(family="teaser")
    with pytest.raises(ValueError, match="unknown dependency"):
        _primitive_contract(dependency_game_ids=("2025_99_FAKE_GAME",))
    with pytest.raises(ValueError, match="kind"):
        _primitive_contract(
            family="same_game_composite",
            kind="primitive",
        )


def test_cross_game_inventory_can_never_be_analysis_eligible() -> None:
    dependencies = ("2025_14_DAL_DET", "2025_14_PHI_LAC")

    with pytest.raises(ValueError, match="analysis_eligible"):
        _primitive_contract(
            family="cross_game_mve_inventory",
            dependency_game_ids=dependencies,
            kind="cross_game_inventory",
            analysis_eligible=True,
        )
    contract = _primitive_contract(
        family="cross_game_mve_inventory",
        dependency_game_ids=dependencies,
        kind="cross_game_inventory",
        analysis_eligible=False,
    )
    assert contract.dependency_game_ids == dependencies
    assert contract.analysis_eligible is False


def test_inventory_rejects_duplicate_contracts_and_wrong_game_dependency() -> None:
    contract = _primitive_contract()

    with pytest.raises(ValueError, match="duplicate contract"):
        spec.ContractInventoryV1(
            native_game_id="2025_14_DAL_DET",
            contracts=(contract, contract),
            exclusions=(),
            unknowns=(),
        )
    with pytest.raises(ValueError, match="inventory game"):
        spec.ContractInventoryV1(
            native_game_id="2025_14_PHI_LAC",
            contracts=(contract,),
            exclusions=(),
            unknowns=(),
        )


def test_observation_requires_decimal_interval_and_valid_book() -> None:
    observed = _observation()

    assert isinstance(observed.price, Decimal)
    assert isinstance(observed.size, Decimal)
    with pytest.raises(TypeError, match="Decimal"):
        _observation(price=0.51)
    with pytest.raises(ValueError, match="interval"):
        _observation(
            source_interval_end=datetime(2025, 12, 5, 1, tzinfo=UTC)
        )
    with pytest.raises(ValueError, match="bid"):
        _observation(bid=Decimal("0.53"), ask=Decimal("0.52"))


def test_quote_observation_does_not_require_a_fabricated_trade_price() -> None:
    quote = _observation(
        observation_kind="kalshi_1m_candle_bbo",
        price=None,
    )

    assert quote.price is None
    assert quote.bid == Decimal("0.50")
    assert quote.ask == Decimal("0.52")
    with pytest.raises(ValueError, match="trade.*price"):
        _observation(price=None)
    with pytest.raises(ValueError, match="candle.*point price"):
        _observation(observation_kind="kalshi_1m_candle_bbo")
    with pytest.raises(ValueError, match="candle.*bid and ask"):
        _observation(
            observation_kind="kalshi_1m_candle_bbo",
            price=None,
            ask=None,
        )
    with pytest.raises(ValueError, match="price, bid, or ask"):
        _observation(
            observation_kind="quote",
            price=None,
            bid=None,
            ask=None,
        )


def test_canonical_json_and_hash_are_stable_float_free_and_sensitive() -> None:
    payload = (
        ("at", datetime(2025, 12, 5, 1, tzinfo=UTC)),
        ("price", Decimal("0.1000")),
        ("values", (1, 2, 3)),
    )

    first = spec.canonical_json_bytes(payload)
    second = spec.canonical_json_bytes(payload)
    parsed = json.loads(first)
    assert first == second
    assert parsed == {
        "at": "2025-12-05T01:00:00Z",
        "price": "0.1",
        "values": [1, 2, 3],
    }
    assert spec.canonical_sha256(payload) == spec.canonical_sha256(payload)
    assert spec.canonical_sha256(payload) != spec.canonical_sha256(
        replace_pair(payload, "price", Decimal("0.2"))
    )
    with pytest.raises(TypeError, match="binary float"):
        spec.canonical_json_bytes((("price", 0.1),))


def replace_pair(
    pairs: tuple[tuple[str, object], ...],
    key: str,
    value: object,
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (pair_key, value if pair_key == key else pair_value)
        for pair_key, pair_value in pairs
    )


def test_plan_and_batch_ids_are_deterministic_and_input_sensitive() -> None:
    batch_spec = spec.X13_BATCH_SPEC
    assert batch_spec.plan_id == spec.canonical_sha256(
        batch_spec.canonical_payload
    )
    assert batch_spec.plan_id == spec.X13_BATCH_SPEC.plan_id

    manifest = spec.BatchManifestV1(
        plan_id=batch_spec.plan_id,
        raw_manifest_sha256s=(_digest("b"),),
        builder_versions=(("builder", "v1"),),
        game_artifact_sha256s=_game_artifact_sha256s(),
        published_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert manifest.batch_id == spec.canonical_sha256(
        manifest.canonical_payload
    )
    assert manifest.batch_id == replace(manifest).batch_id
    changed = replace(
        manifest,
        builder_versions=(("builder", "v2"),),
    )
    assert changed.batch_id != manifest.batch_id


def test_plan_id_binds_bindings_catalog_and_frozen_source_windows() -> None:
    batch_spec = spec.X13_BATCH_SPEC
    assert batch_spec.native_game_bindings_sha256 == spec.canonical_sha256(
        spec.X13_GAME_BINDINGS
    )
    assert batch_spec.venue_family_catalog_sha256 == spec.canonical_sha256(
        spec.X13_VENUE_FAMILY_CATALOG
    )
    assert batch_spec.source_window_universe_id == (
        "sha256:01526155f3436a752cd6765d5b6a88086ab71badac84e82518a526cc8a7098d2"
    )

    changed_binding = replace(
        spec.X13_GAME_BINDINGS[0],
        official_reference_status="NARROW_FACT_CHECK_ONLY",
    )
    changed_bindings = (changed_binding, *spec.X13_GAME_BINDINGS[1:])
    assert spec.build_x13_batch_spec(
        native_game_bindings=changed_bindings,
        venue_family_catalog=spec.X13_VENUE_FAMILY_CATALOG,
        source_window_universe_id=spec.X13_SOURCE_WINDOW_UNIVERSE_ID,
    ).plan_id != batch_spec.plan_id

    changed_catalog = replace(
        spec.X13_VENUE_FAMILY_CATALOG,
        primitive_families=tuple(
            reversed(spec.X13_VENUE_FAMILY_CATALOG.primitive_families)
        ),
    )
    assert spec.build_x13_batch_spec(
        native_game_bindings=spec.X13_GAME_BINDINGS,
        venue_family_catalog=changed_catalog,
        source_window_universe_id=spec.X13_SOURCE_WINDOW_UNIVERSE_ID,
    ).plan_id != batch_spec.plan_id

    assert spec.build_x13_batch_spec(
        native_game_bindings=spec.X13_GAME_BINDINGS,
        venue_family_catalog=spec.X13_VENUE_FAMILY_CATALOG,
        source_window_universe_id=_digest("f"),
    ).plan_id != batch_spec.plan_id


def test_canonical_pair_fields_are_tuples_sorted_and_reject_float() -> None:
    plan = spec.CapturePlanV1(
        plan_id=spec.X13_BATCH_SPEC.plan_id,
        native_game_ids=spec.X13_GAME_IDS,
        contract_ids=("contract-1",),
        source_windows=(("post_seconds", 60), ("pre_seconds", 30)),
        required_free_bytes=1,
    )
    receipt = spec.RawCaptureReceiptV1(
        source="kalshi",
        request_identity=(("event_ticker", "KXNFLGAME-25DEC04DALDET"),),
        object_sha256=_digest("d"),
        manifest_sha256=_digest("e"),
        byte_length=1,
        terminal_proof=(("cursor", "terminal"),),
    )
    candidate = spec.AssociationCandidateV1(
        episode_id="episode-1",
        contract_id="contract-1",
        delay_scenario_seconds=0,
        horizon_seconds=1,
        status="PRELIMINARY_SOURCE_TIME_ONLY",
        order_ambiguous=False,
        contaminated=False,
        metrics=(("delta", Decimal("-0.1")),),
    )

    assert isinstance(plan.source_windows, tuple)
    assert isinstance(receipt.request_identity, tuple)
    assert isinstance(receipt.terminal_proof, tuple)
    assert isinstance(candidate.metrics, tuple)
    with pytest.raises(TypeError, match="binary float"):
        replace(candidate, metrics=(("delta", -0.1),))
    with pytest.raises(ValueError, match="sorted"):
        replace(plan, source_windows=(("pre_seconds", 30), ("post_seconds", 60)))
    with pytest.raises(ValueError, match="contract_ids"):
        replace(plan, contract_ids=())


def test_capture_plan_and_manifest_require_exact_frozen_twenty() -> None:
    with pytest.raises(ValueError, match="exactly 20"):
        spec.CapturePlanV1(
            plan_id=spec.X13_BATCH_SPEC.plan_id,
            native_game_ids=spec.X13_GAME_IDS[:-1],
            contract_ids=(),
            source_windows=(),
            required_free_bytes=1,
        )
    with pytest.raises(ValueError, match="exactly 20"):
        spec.BatchManifestV1(
            plan_id=spec.X13_BATCH_SPEC.plan_id,
            raw_manifest_sha256s=(_digest("b"),),
            builder_versions=(("builder", "v1"),),
            game_artifact_sha256s=_game_artifact_sha256s()[:-1],
        )


def test_all_public_records_are_frozen_slot_dataclasses() -> None:
    records = (
        spec.BatchSpecV1,
        spec.NativeGameBindingV1,
        spec.VenueFamilyCatalogV1,
        spec.ContractInventoryV1,
        spec.CapturePlanV1,
        spec.RawCaptureReceiptV1,
        spec.MarketContractV1,
        spec.MarketObservationV1,
        spec.AssociationCandidateV1,
        spec.BatchManifestV1,
    )

    for record in records:
        assert is_dataclass(record)
        assert record.__dataclass_params__.frozen is True
        assert hasattr(record, "__slots__")
