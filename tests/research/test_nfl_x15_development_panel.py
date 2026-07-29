from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market.research import nfl_x15_development_panel as _development_panel
from prediction_market.research.nfl_x15_development_panel import (
    CONFIRMATORY_CLAIM_BOUNDARY,
    DIAGNOSTIC_CLAIM_BOUNDARY,
    DevelopmentPanelError,
    DevelopmentSourceSpec,
    VenueConfirmatoryEvidence,
    publish_exact153_development_panel,
)


GAME_ID = "2025_01_AWY_HME"
AUTHORITY_SHA = "sha256:" + "a" * 64
SOURCE_SHA = "sha256:" + "b" * 64
REGISTRY_SHA = "sha256:" + "c" * 64
FACTOR_REGISTRY = {
    "factor_count": 59,
    "factor_identity_sha256": "sha256:" + "e" * 64,
    "file_sha256": "sha256:" + "f" * 64,
    "schema": "NFLFactorRegistryV4",
    "semantic_sha256": REGISTRY_SHA,
    "status": "AUTHORITATIVE",
    "version": "v4",
}


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _evidence_frame_sha256(frame: pd.DataFrame) -> str:
    columns = sorted(map(str, frame.columns))
    rows = frame.loc[:, columns].to_dict("records")
    rows.sort(key=_development_panel._canonical_bytes)
    return _development_panel._canonical_sha256(
        {
            "schema": "nfl_x15_confirmatory_evidence_frame_v1",
            "columns": columns,
            "rows": rows,
        }
    )


def _publish_parquet(
    root: Path,
    relative_prefix: str,
    frame: pd.DataFrame,
    *,
    descriptor_root: str | None = None,
) -> dict:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), sink)
    payload = sink.getvalue().to_pybytes()
    digest = _sha(payload)
    hexadecimal = digest.removeprefix("sha256:")
    relative = (
        Path(relative_prefix)
        / "objects"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.parquet"
    )
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return {
        "object_path": relative.relative_to(
            descriptor_root or relative_prefix
        ).as_posix(),
        "object_sha256": digest,
        "byte_length": len(payload),
        "row_count": len(frame),
        "schema_columns": list(frame.columns),
        "schema_fingerprint": "sha256:" + "d" * 64,
    }


def _publish_game_manifest(
    root: Path,
    relative_prefix: str,
    *,
    game_id: str,
    schema: str,
    tables: dict[str, pd.DataFrame],
    market_style: bool,
    descriptor_root: str | None = None,
    object_descriptor_root: str | None = None,
    publication_id: str | None = None,
    factor_registry: dict[str, object] | None = None,
) -> dict:
    descriptors = {
        name: _publish_parquet(
            root,
            relative_prefix,
            frame,
            descriptor_root=object_descriptor_root or descriptor_root,
        )
        for name, frame in tables.items()
    }
    material: dict[str, object] = {
        "schema": schema,
        "game_id": game_id,
        "cohort": "development",
        "publication_gate": "PASS",
        "holdout_reaction_accessed": False,
    }
    if publication_id is not None:
        material["publication_id"] = publication_id
    if factor_registry is not None:
        material["factor_registry"] = factor_registry
        material["counts"] = {"factor_coverage_audit_rows": 59}
    if market_style:
        material["stages"] = descriptors
    else:
        material["tables"] = [
            {"name": name, **descriptor}
            for name, descriptor in descriptors.items()
        ]
    material["bundle_sha256"] = _sha(_canonical(material))
    payload = _canonical(material)
    digest = _sha(payload)
    hexadecimal = digest.removeprefix("sha256:")
    relative = (
        Path(relative_prefix)
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.manifest.json"
    )
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return {
        "game_id": game_id,
        "manifest_path": relative.relative_to(
            descriptor_root or relative_prefix
        ).as_posix(),
        "manifest_sha256": digest,
        "bundle_sha256": material["bundle_sha256"],
        **(
            {"counts": {"factor_coverage_audit_rows": 59}}
            if factor_registry is not None
            else {}
        ),
    }


def _publish_batch(
    root: Path,
    relative_prefix: str,
    *,
    schema: str,
    game: dict,
    market_style: bool,
    publication_id: str | None = None,
    factor_registry: dict[str, object] | None = None,
) -> tuple[Path, str]:
    material: dict[str, object] = {
        "schema": schema,
        "experiment_id": "X-13" if "stage_a" not in schema else "X-15",
        "cohort": "development",
        "game_count": 1,
        "games": [game],
        "publication_gate": "PASS",
    }
    if publication_id is not None:
        material["publication_id"] = publication_id
    if factor_registry is not None:
        material["factor_registry"] = factor_registry
    if market_style:
        material["final_holdout_access"] = "CLOSED"
    else:
        material["holdout_reaction_accessed"] = False
        material["market_data_read"] = False
    material["batch_sha256"] = _sha(_canonical(material))
    payload = _canonical(material)
    digest = _sha(payload)
    hexadecimal = digest.removeprefix("sha256:")
    relative = (
        Path(relative_prefix)
        / "batches"
        / "manifests"
        / "sha256"
        / hexadecimal[:2]
        / f"{hexadecimal}.batch-index.json"
    )
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target, digest


def _facts() -> tuple[pd.DataFrame, pd.DataFrame]:
    facts = pd.DataFrame(
        [
            {
                "game_id": GAME_ID,
                "event_id": "event-1",
                "play_id": "play-1",
                "atomic_information_episode_id": "episode-1",
                "source_interval_start": "2025-09-07T12:00:00Z",
                "source_interval_end": "2025-09-07T12:00:01Z",
                "known_at": "2025-09-07T12:00:01Z",
                "source_resolution": "1s",
                "stage_b_information_event_eligible": True,
                "home_team": "HME",
                "away_team": "AWY",
                "outcome_tags": '["PASS_COMPLETE"]',
                "pbp_source_sha256": SOURCE_SHA,
                "game_seconds_remaining": 3300,
                "score_margin_home": 0,
                "possession_is_home": True,
                "down": 1,
                "distance": 10,
                "yardline_100": 65.0,
                "primary_action": "PASS",
                "yards_gained": 12.0,
                "return_yards": 0.0,
                "actor_is_home": True,
                "beneficiary_is_home": True,
            },
            {
                "game_id": GAME_ID,
                "event_id": "event-2",
                "play_id": "play-2",
                "atomic_information_episode_id": "episode-2",
                "source_interval_start": "2025-09-07T12:02:00Z",
                "source_interval_end": "2025-09-07T12:02:01Z",
                "known_at": "2025-09-07T12:02:01Z",
                "source_resolution": "1s",
                "stage_b_information_event_eligible": True,
                "home_team": "HME",
                "away_team": "AWY",
                "outcome_tags": '["GAME_END"]',
                "pbp_source_sha256": SOURCE_SHA,
                "game_seconds_remaining": 0,
                "score_margin_home": 7,
                "possession_is_home": False,
                "down": pd.NA,
                "distance": pd.NA,
                "yardline_100": pd.NA,
                "primary_action": "GAME_END",
                "yards_gained": 0.0,
                "return_yards": 0.0,
                "actor_is_home": False,
                "beneficiary_is_home": True,
            },
        ]
    )
    hits = pd.DataFrame(
        [
            {
                "game_id": GAME_ID,
                "event_id": "event-1",
                "play_id": "play-1",
                "factor_id": "NFL.EVENT.PASS_COMPLETE",
                "factor_version": "v3",
                "registry_sha256": REGISTRY_SHA,
                "pbp_source_sha256": SOURCE_SHA,
                "predicate_evidence": '{"complete":true}',
            }
        ]
    )
    return facts, hits


def _references() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": GAME_ID,
                "event_id": "event-1",
                "atomic_information_episode_id": "episode-1",
                "pre_state_known_at": "2025-09-07T11:59:59Z",
                "post_state_known_at": "2025-09-07T12:00:01Z",
                "p_before_home": 0.50,
                "p_after_home": 0.56,
                "reference_delta_home": 0.06,
                "reference_status": "SUPPORTED",
            },
            {
                "game_id": GAME_ID,
                "event_id": "event-2",
                "atomic_information_episode_id": "episode-2",
                "pre_state_known_at": "2025-09-07T12:01:59Z",
                "post_state_known_at": "2025-09-07T12:02:01Z",
                "p_before_home": 0.95,
                "p_after_home": 1.0,
                "reference_delta_home": 0.05,
                "reference_status": "SUPPORTED",
            },
        ]
    )


def _market() -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory_rows = []
    observations = []
    for venue in ("polymarket", "kalshi"):
        logical = f"{venue}:{GAME_ID}:winner"
        home_contract = "poly-home-token" if venue == "polymarket" else "KX-HME"
        away_contract = "poly-away-token" if venue == "polymarket" else "KX-AWY"
        raw_market = "poly-condition" if venue == "polymarket" else home_contract
        for outcome, raw_contract in (
            ("HME", home_contract),
            ("AWY", away_contract),
        ):
            inventory_rows.append(
                {
                    "game_id": GAME_ID,
                    "logical_market_id": logical,
                    "outcome": outcome,
                    "venue": venue,
                    "contract_id": "ambiguous-logical-contract",
                    "venue_market_id": "venue-market",
                    "raw_contract_id": raw_contract,
                    "family": "moneyline",
                    "period": "full_game",
                    "subject": GAME_ID,
                    "measure": "winner",
                    "kind": "primitive",
                    "analysis_eligible": True,
                    "rule_sha256": SOURCE_SHA,
                }
            )
        for second in range(2, 63):
            source_time = (
                pd.Timestamp("2025-09-07T12:00:00Z")
                + pd.Timedelta(seconds=second)
            )
            observations.append(
                {
                    "game_id": GAME_ID,
                    "observation_id": f"{venue}-home-{second}",
                    "venue": venue,
                    "raw_market_id": raw_market,
                    "logical_market_id": logical,
                    "outcome": "HME",
                    "kind": "trade",
                    "source_start_utc": source_time.isoformat(),
                    "source_end_utc": (
                        source_time + pd.Timedelta(microseconds=1)
                    ).isoformat(),
                    "source_end_inclusive": False,
                    "native_source_time_utc": source_time.isoformat(),
                    "price": 0.50 + second / 1000,
                    "size": 10.0,
                    "provenance": "observed",
                    "native_id": f"native-{venue}-{second}",
                    "primary_path_eligible": True,
                }
            )
            if second == 2:
                observations.append(
                    {
                        **observations[-1],
                        "observation_id": f"{venue}-home-{second}-second-fill",
                        "native_id": f"native-{venue}-{second}-second-fill",
                        "price": 0.60,
                        "size": 5.0,
                    }
                )
    return pd.DataFrame(observations), pd.DataFrame(inventory_rows)


def _source_fixture(
    tmp_path: Path,
    *,
    facts_override: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    references_override: pd.DataFrame | None = None,
    facts_batch_schema: str = "nfl_x13_exact153_fact_batch_index_v4",
    facts_game_schema: str = (
        "nfl_x13_exact153_single_game_fact_manifest_v4"
    ),
    batch_publication_id: str | None = "exact-153-facts-v4",
    game_publication_id: str | None = "exact-153-facts-v4",
    batch_factor_registry: dict[str, object] | None = FACTOR_REGISTRY,
    game_factor_registry: dict[str, object] | None = FACTOR_REGISTRY,
    facts_descriptor_root: str = "x13",
    facts_object_descriptor_root: str | None = None,
) -> DevelopmentSourceSpec:
    facts, hits = _facts() if facts_override is None else facts_override
    observations, inventory = _market()
    facts_dataset = "x13/exact-153-facts-v4"
    facts_game_root = f"{facts_dataset}/single-game/{GAME_ID}"
    facts_game = _publish_game_manifest(
        tmp_path,
        facts_game_root,
        game_id=GAME_ID,
        schema=facts_game_schema,
        tables={"canonical_factor_events": facts, "factor_hits": hits},
        market_style=False,
        descriptor_root=facts_descriptor_root,
        object_descriptor_root=facts_object_descriptor_root,
        publication_id=game_publication_id,
        factor_registry=game_factor_registry,
    )
    refs_game = _publish_game_manifest(
        tmp_path,
        "stage-a",
        game_id=GAME_ID,
        schema="nfl_x15_stage_a_single_game_manifest_v1",
        tables={
            "reference_observations": (
                _references()
                if references_override is None
                else references_override
            )
        },
        market_style=False,
    )
    market_game = _publish_game_manifest(
        tmp_path,
        "market",
        game_id=GAME_ID,
        schema="nfl_expansion_development_market_game_manifest_v1",
        tables={
            "actual_market_observations": observations,
            "contract_inventory": inventory,
        },
        market_style=True,
    )
    facts_batch, facts_sha = _publish_batch(
        tmp_path,
        facts_dataset,
        schema=facts_batch_schema,
        game=facts_game,
        market_style=False,
        publication_id=batch_publication_id,
        factor_registry=batch_factor_registry,
    )
    refs_batch, refs_sha = _publish_batch(
        tmp_path,
        "stage-a",
        schema="nfl_x15_stage_a_batch_index_v1",
        game=refs_game,
        market_style=False,
    )
    market_batch, market_sha = _publish_batch(
        tmp_path,
        "market",
        schema="nfl_expansion_development_market_batch_index_v1",
        game=market_game,
        market_style=True,
    )
    authority = {
        "schema": "nfl_factor_expansion_registry_v2",
        "split_lock": {
            "development": {"game_count": 1, "weeks": [1]},
            "final_holdout": {"game_count": 0, "weeks": []},
            "game_assignments": [
                {"cohort": "development", "game_id": GAME_ID, "week": 1}
            ],
        },
    }
    authority_payload = _canonical(authority)
    authority_sha = _sha(authority_payload)
    authority_object = tmp_path / "authority.json"
    authority_object.write_bytes(authority_payload)
    authority_manifest = {
        "schema": "nfl_factor_expansion_registry_manifest_v2",
        "object_path": "authority.json",
        "object_sha256": authority_sha,
        "byte_length": len(authority_payload),
        "development_game_count": 1,
        "final_holdout_game_count": 0,
    }
    authority_manifest_payload = _canonical(authority_manifest)
    authority_manifest_path = tmp_path / "authority.manifest.json"
    authority_manifest_path.write_bytes(authority_manifest_payload)
    return DevelopmentSourceSpec(
        facts_batch_path=facts_batch,
        facts_batch_file_sha256=facts_sha,
        stage_a_batch_path=refs_batch,
        stage_a_batch_file_sha256=refs_sha,
        market_batch_path=market_batch,
        market_batch_file_sha256=market_sha,
        authority_manifest_path=authority_manifest_path,
        authority_manifest_file_sha256=_sha(authority_manifest_payload),
        authority_object_path=authority_object,
        authority_object_sha256=authority_sha,
        expected_game_count=1,
    )


def test_default_facts_source_pins_exact_v4_artifact() -> None:
    expected_path = (
        "artifacts/market-observation/nfl/x13/exact-153-facts-v4/"
        "batches/manifests/sha256/5d/"
        "5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757"
        ".batch-index.json"
    )
    expected_sha = (
        "sha256:"
        "5d693723e991b7f691dab2826308773a0ce6a30564c37dcb7d4a1cb9e1580757"
    )

    assert _development_panel.DEFAULT_FACTS_BATCH.as_posix() == expected_path
    assert _development_panel.DEFAULT_FACTS_BATCH_FILE_SHA256 == expected_sha
    default_spec = _development_panel.default_development_source_spec()
    assert default_spec.facts_batch_path.as_posix() == expected_path
    assert default_spec.facts_batch_file_sha256 == expected_sha


def test_v4_facts_paths_resolve_from_x13_parent_without_changing_v1_roots(
    tmp_path: Path,
) -> None:
    spec = _source_fixture(tmp_path)

    verified = _development_panel.verify_development_sources(
        project_root=tmp_path,
        source_spec=spec,
    )
    manifest, manifest_path = _development_panel._verify_game_manifest(
        batch=verified.facts,
        descriptor=verified.facts.games[GAME_ID],
        label="X13 facts",
    )
    facts = _development_panel._read_verified_table(
        batch=verified.facts,
        manifest=manifest,
        table_name="canonical_factor_events",
        market_style=False,
        label=f"X13 facts.{GAME_ID}",
    )

    assert verified.facts.root == (tmp_path / "x13").resolve()
    assert manifest_path.is_relative_to(
        tmp_path / "x13" / "exact-153-facts-v4" / "single-game" / GAME_ID
    )
    assert verified.stage_a.root == (tmp_path / "stage-a").resolve()
    assert verified.market.root == (tmp_path / "market").resolve()
    assert facts["game_id"].tolist() == [GAME_ID, GAME_ID]


def test_v1_facts_batch_fails_closed(tmp_path: Path) -> None:
    spec = _source_fixture(
        tmp_path,
        facts_batch_schema="nfl_x13_exact153_fact_batch_index_v1",
    )

    with pytest.raises(DevelopmentPanelError, match="X13 facts batch contract"):
        _development_panel.verify_development_sources(
            project_root=tmp_path,
            source_spec=spec,
        )


@pytest.mark.parametrize(
    "fixture_kwargs",
    [
        {"batch_publication_id": "exact-153-facts-v3"},
        {
            "batch_factor_registry": {
                **FACTOR_REGISTRY,
                "factor_count": 58,
            }
        },
    ],
)
def test_v4_facts_batch_rejects_wrong_publication_or_registry(
    tmp_path: Path,
    fixture_kwargs: dict[str, object],
) -> None:
    spec = _source_fixture(tmp_path, **fixture_kwargs)

    with pytest.raises(DevelopmentPanelError, match="X13 facts batch contract"):
        _development_panel.verify_development_sources(
            project_root=tmp_path,
            source_spec=spec,
        )


@pytest.mark.parametrize(
    "fixture_kwargs",
    [
        {
            "facts_game_schema": (
                "nfl_x13_exact153_fact_single_game_manifest_v1"
            )
        },
        {"game_publication_id": "exact-153-facts-v3"},
        {
            "game_factor_registry": {
                **FACTOR_REGISTRY,
                "semantic_sha256": SOURCE_SHA,
            }
        },
    ],
)
def test_v4_game_manifest_rejects_wrong_schema_publication_or_registry_binding(
    tmp_path: Path,
    fixture_kwargs: dict[str, object],
) -> None:
    spec = _source_fixture(tmp_path, **fixture_kwargs)
    verified = _development_panel.verify_development_sources(
        project_root=tmp_path,
        source_spec=spec,
    )

    with pytest.raises(DevelopmentPanelError, match="manifest contract"):
        _development_panel._verify_game_manifest(
            batch=verified.facts,
            descriptor=verified.facts.games[GAME_ID],
            label="X13 facts",
        )


@pytest.mark.parametrize(
    "fixture_kwargs",
    [
        {"facts_descriptor_root": "x13/exact-153-facts-v4"},
        {"facts_object_descriptor_root": "x13/exact-153-facts-v4"},
    ],
)
def test_v4_facts_rejects_dataset_relative_v1_path_shapes(
    tmp_path: Path,
    fixture_kwargs: dict[str, object],
) -> None:
    spec = _source_fixture(tmp_path, **fixture_kwargs)

    with pytest.raises(DevelopmentPanelError, match="V4 path contract"):
        publish_exact153_development_panel(
            project_root=tmp_path,
            output_root=tmp_path / "published",
            source_spec=spec,
            confirmatory_evidence={},
        )


def test_factor_hits_must_bind_v4_registry_semantic_sha(tmp_path: Path) -> None:
    facts, hits = _facts()
    hits["registry_sha256"] = SOURCE_SHA
    spec = _source_fixture(
        tmp_path,
        facts_override=(facts, hits),
    )

    with pytest.raises(DevelopmentPanelError, match="factor_hits registry binding"):
        publish_exact153_development_panel(
            project_root=tmp_path,
            output_root=tmp_path / "published",
            source_spec=spec,
            confirmatory_evidence={},
        )


def test_stage_a_must_join_every_eligible_fact_episode(tmp_path: Path) -> None:
    references = _references().loc[
        lambda frame: frame["atomic_information_episode_id"].ne("episode-2")
    ]
    spec = _source_fixture(
        tmp_path,
        references_override=references,
    )

    with pytest.raises(DevelopmentPanelError, match="Stage A game/episode join"):
        publish_exact153_development_panel(
            project_root=tmp_path,
            output_root=tmp_path / "published",
            source_spec=spec,
            confirmatory_evidence={},
        )


def test_landmark_uses_next_fact_finalized_known_at_boundary(
    tmp_path: Path,
) -> None:
    facts, hits = _facts()
    second = facts["atomic_information_episode_id"].eq("episode-2")
    facts.loc[second, "source_interval_start"] = "2025-09-07T12:00:03Z"
    facts.loc[second, "source_interval_end"] = "2025-09-07T12:00:04Z"
    facts.loc[second, "known_at"] = "2025-09-07T12:00:04Z"
    spec = _source_fixture(
        tmp_path / "source",
        facts_override=(facts, hits),
    )

    publication = publish_exact153_development_panel(
        project_root=tmp_path / "source",
        output_root=tmp_path / "source" / "published",
        source_spec=spec,
        confirmatory_evidence={},
    )
    _, tables = _read_game_tables(publication)
    panel = tables["diagnostic_panel"]
    invalid = panel.loc[
        panel["atomic_information_episode_id"].eq("episode-1")
        & panel["landmark_seconds"].ge(3)
    ]
    before_finalization = panel.loc[
        panel["atomic_information_episode_id"].eq("episode-1")
        & panel["landmark_seconds"].eq(2)
    ]

    assert set(panel["schema_version"]) == {
        "HistoricalTradesOnlyProbabilityPanelV2"
    }
    assert before_finalization["sports_clean_l"].eq(True).all()
    assert set(before_finalization["sports_clean_l_reason"]) == {
        "SPORTS_WINDOW_CLEAN"
    }
    assert before_finalization["decision_eligible"].eq(True).all()
    assert invalid["sports_clean_l"].eq(False).all()
    assert set(invalid["sports_clean_l_reason"]) == {
        "NEXT_FINALIZED_INFORMATION_EVENT_AT_OR_BEFORE_L"
    }
    assert invalid["decision_eligible"].eq(False).all()
    assert set(invalid["attrition_reason"]) == {
        "LANDMARK_NEXT_FINALIZED_INFORMATION_EVENT_AT_OR_BEFORE_L"
    }


def test_unknown_terminal_continuity_fails_closed_with_explicit_reason(
    tmp_path: Path,
) -> None:
    facts, hits = _facts()
    terminal = facts["atomic_information_episode_id"].eq("episode-2")
    facts.loc[terminal, "outcome_tags"] = "[]"
    facts.loc[terminal, "primary_action"] = "UNKNOWN_TERMINAL_STATUS"
    facts.loc[terminal, "game_seconds_remaining"] = 1
    facts.loc[terminal, "source_interval_start"] = "2025-09-07T12:00:03Z"
    facts.loc[terminal, "source_interval_end"] = "2025-09-07T12:00:04Z"
    facts.loc[terminal, "known_at"] = "2025-09-07T12:00:04Z"
    spec = _source_fixture(
        tmp_path / "source",
        facts_override=(facts, hits),
    )

    publication = publish_exact153_development_panel(
        project_root=tmp_path / "source",
        output_root=tmp_path / "source" / "published",
        source_spec=spec,
        confirmatory_evidence={},
    )
    _, tables = _read_game_tables(publication)
    final_episode = tables["diagnostic_panel"].loc[
        tables["diagnostic_panel"][
            "atomic_information_episode_id"
        ].eq("episode-2")
    ]

    assert not final_episode.empty
    assert final_episode["sports_clean_l"].eq(False).all()
    assert set(final_episode["sports_clean_l_reason"]) == {
        "SPORTS_CONTINUITY_UNKNOWN"
    }
    assert final_episode["decision_eligible"].eq(False).all()
    assert set(final_episode["attrition_reason"]) == {
        "LANDMARK_SPORTS_CONTINUITY_UNKNOWN"
    }


def _confirmatory_evidence(venue: str) -> VenueConfirmatoryEvidence:
    tick_rules = pd.DataFrame(
        [
            {
                "venue": venue,
                "tick_rule_id": f"{venue}-historical-rule",
                "effective_start_utc": "2025-09-07T00:00:00Z",
                "effective_end_utc": "2025-09-08T00:00:00Z",
                "tick_size": 0.01,
            }
        ]
    )
    market_continuity = pd.DataFrame(
        [
            {
                "game_id": GAME_ID,
                "atomic_information_episode_id": episode,
                "continuity_verified_until_utc": (
                    "2025-09-07T12:01:05Z"
                    if episode == "episode-1"
                    else "2025-09-07T12:03:05Z"
                ),
                "suspension_time_utc": pd.NaT,
                "continuity_gap_time_utc": pd.NaT,
            }
            for episode in ("episode-1", "episode-2")
        ]
    )
    return VenueConfirmatoryEvidence(
        venue=venue,
        tick_rule_id=f"{venue}-historical-rule",
        tick_rules=tick_rules,
        market_continuity=market_continuity,
        tick_rule_source_sha256=_evidence_frame_sha256(tick_rules),
        continuity_source_sha256=_evidence_frame_sha256(market_continuity),
    )


def _read_game_tables(publication) -> tuple[dict, dict[str, pd.DataFrame]]:
    game = publication.games[0]
    manifest = json.loads(game.manifest_path.read_text())
    tables = {}
    for descriptor in manifest["tables"]:
        tables[descriptor["name"]] = pd.read_parquet(
            publication.output_root / descriptor["object_path"]
        )
    return manifest, tables


def test_complete_evidence_calls_real_v3_builder_and_keeps_diagnostic_separate(
    tmp_path: Path,
) -> None:
    spec = _source_fixture(tmp_path / "source")
    evidence = {
        (GAME_ID, venue): _confirmatory_evidence(venue)
        for venue in ("polymarket", "kalshi")
    }

    publication = publish_exact153_development_panel(
        project_root=tmp_path / "source",
        output_root=tmp_path / "source" / "published",
        source_spec=spec,
        confirmatory_evidence=evidence,
    )
    manifest, tables = _read_game_tables(publication)

    assert publication.game_count == 1
    assert publication.cohort_mapping_sha256.startswith("sha256:")
    assert set(tables["source_evidence_audit"]["confirmatory_status"]) == {"PASS"}
    assert set(tables["confirmatory_panel"]["schema_version"]) == {
        "VenueReactionPanelV3"
    }
    assert set(tables["confirmatory_panel"]["claim_boundary"]) == {
        CONFIRMATORY_CLAIM_BOUNDARY
    }
    assert set(tables["diagnostic_panel"]["schema_version"]) == {
        "HistoricalTradesOnlyProbabilityPanelV2"
    }
    assert set(tables["diagnostic_panel"]["claim_boundary"]) == {
        DIAGNOSTIC_CLAIM_BOUNDARY
    }
    assert set(tables["diagnostic_panel"]["target_contract"]) == {
        "HISTORICAL_TRADES_ONLY_HOME_PROBABILITY"
    }
    assert set(tables["diagnostic_panel"]["direction_threshold_probability"]) == {
        0.01
    }
    assert set(tables["diagnostic_panel"]["market_continuity_support"]) == {
        "UNKNOWN"
    }
    assert set(tables["diagnostic_panel"]["actual_home_contract_id"]) == {
        "poly-home-token",
        "KX-HME",
    }
    assert "ambiguous-logical-contract" not in set(
        tables["diagnostic_panel"]["actual_home_contract_id"]
    )
    diagnostic_l1 = tables["diagnostic_panel"].loc[
        tables["diagnostic_panel"]["landmark_seconds"].eq(1)
        & tables["diagnostic_panel"]["atomic_information_episode_id"].eq(
            "episode-1"
        )
    ]
    assert set(diagnostic_l1["mark_l_semantics"]) == {
        "LATEST_SOURCE_TIMESTAMP_SIZE_WEIGHTED_VWAP"
    }
    assert set(diagnostic_l1["mark_l_observation_count"]) == {2}
    assert set(diagnostic_l1["mark_l_observed_size"]) == {15.0}
    assert diagnostic_l1["mark_l_trade_id_set_sha256"].str.match(
        r"sha256:[0-9a-f]{64}"
    ).all()
    assert diagnostic_l1["mark_l_price"].round(8).eq(
        round((0.502 * 10 + 0.60 * 5) / 15, 8)
    ).all()
    confirmatory_l1 = tables["confirmatory_panel"].loc[
        tables["confirmatory_panel"]["landmark_seconds"].eq(1)
        & tables["confirmatory_panel"]["atomic_information_episode_id"].eq(
            "episode-1"
        )
    ]
    assert confirmatory_l1["mark_l_price"].isna().all()
    assert set(confirmatory_l1["attrition_reason"]) == {
        "LANDMARK_ORDER_AMBIGUOUS"
    }
    assert manifest["confirmatory_venue_count"] == 2


def test_missing_rule_and_continuity_fail_confirmatory_but_publish_diagnostic(
    tmp_path: Path,
) -> None:
    spec = _source_fixture(tmp_path / "source")

    publication = publish_exact153_development_panel(
        project_root=tmp_path / "source",
        output_root=tmp_path / "source" / "published",
        source_spec=spec,
        confirmatory_evidence={},
    )
    manifest, tables = _read_game_tables(publication)

    audit = tables["source_evidence_audit"]
    assert set(audit["confirmatory_status"]) == {"FAIL_CLOSED"}
    assert set(audit["confirmatory_reason"]) == {
        "MISSING_HISTORICAL_TICK_RULE_AND_MARKET_CONTINUITY_EVIDENCE"
    }
    assert "confirmatory_panel" not in tables
    assert len(tables["diagnostic_panel"]) > 0
    assert set(tables["diagnostic_panel"]["venue_tick_support"]) == {"UNSUPPORTED"}
    assert manifest["confirmatory_venue_count"] == 0
    assert manifest["diagnostic_venue_count"] == 2


def test_verified_table_decodes_the_bytes_that_passed_sha_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _publish_parquet(
        tmp_path,
        "dataset",
        pd.DataFrame({"value": [1]}),
    )
    object_path = tmp_path / "dataset" / descriptor["object_path"]
    tampered_payload = _development_panel._parquet_bytes(
        pd.DataFrame({"value": [999]})
    )
    batch = _development_panel._VerifiedBatch(
        path=tmp_path / "dataset" / "batch.json",
        file_sha256=SOURCE_SHA,
        root=tmp_path / "dataset",
        document=MappingProxyType({}),
        games=MappingProxyType({}),
    )
    manifest = {"tables": [{"name": "payload", **descriptor}]}
    stable_read = _development_panel._stable_read

    def _swap_after_verification(path: Path, **kwargs) -> bytes:
        payload = stable_read(path, **kwargs)
        path.write_bytes(tampered_payload)
        return payload

    monkeypatch.setattr(
        _development_panel,
        "_stable_read",
        _swap_after_verification,
    )

    observed = _development_panel._read_verified_table(
        batch=batch,
        manifest=manifest,
        table_name="payload",
        market_style=False,
        label="adversarial",
    )

    assert observed["value"].tolist() == [1]


def test_invented_confirmatory_source_sha_cannot_unlock_evidence() -> None:
    packet = _confirmatory_evidence("kalshi")
    packet = replace(packet, tick_rule_source_sha256=SOURCE_SHA)

    with pytest.raises(
        DevelopmentPanelError,
        match="tick_rule_source_sha256 does not bind canonical evidence",
    ):
        _development_panel._confirmatory_evidence_audit(
            game_id=GAME_ID,
            venues=["kalshi"],
            evidence={(GAME_ID, "kalshi"): packet},
        )


def test_kalshi_observations_require_exact_raw_ticker_identity() -> None:
    observations, inventory = _market()

    with pytest.raises(
        DevelopmentPanelError,
        match="market input missing columns.*raw_market_id",
    ):
        _development_panel._adapt_market(
            game_id=GAME_ID,
            home_team="HME",
            observations=observations.drop(columns=["raw_market_id"]),
            inventory=inventory,
        )


def test_tampered_published_input_fails_before_output(tmp_path: Path) -> None:
    spec = _source_fixture(tmp_path / "source")
    market_batch = spec.market_batch_path
    document = json.loads(market_batch.read_text())
    document["game_count"] = 2
    market_batch.write_bytes(_canonical(document))

    with pytest.raises(DevelopmentPanelError, match="SHA-256 mismatch"):
        publish_exact153_development_panel(
            project_root=tmp_path / "source",
            output_root=tmp_path / "source" / "published",
            source_spec=spec,
            confirmatory_evidence={},
        )


def test_authority_holdout_count_is_exact_not_advisory(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    assignments = [
        {
            "cohort": "development",
            "game_id": f"2025_{(index % 12) + 1:02d}_A{index:03d}_H{index:03d}",
            "week": (index % 12) + 1,
        }
        for index in range(153)
    ]
    authority = {
        "schema": "nfl_factor_expansion_registry_v2",
        "split_lock": {"game_assignments": assignments},
    }
    object_payload = _canonical(authority)
    object_path = source_root / "authority.json"
    object_path.write_bytes(object_payload)
    manifest = {
        "object_sha256": _sha(object_payload),
        "byte_length": len(object_payload),
        "development_game_count": 153,
        "final_holdout_game_count": 0,
    }
    manifest_payload = _canonical(manifest)
    manifest_path = source_root / "authority.manifest.json"
    manifest_path.write_bytes(manifest_payload)
    spec = DevelopmentSourceSpec(
        facts_batch_path=tmp_path / "unused-facts.json",
        facts_batch_file_sha256=SOURCE_SHA,
        stage_a_batch_path=tmp_path / "unused-stage-a.json",
        stage_a_batch_file_sha256=SOURCE_SHA,
        market_batch_path=tmp_path / "unused-market.json",
        market_batch_file_sha256=SOURCE_SHA,
        authority_manifest_path=manifest_path,
        authority_manifest_file_sha256=_sha(manifest_payload),
        authority_object_path=object_path,
        authority_object_sha256=_sha(object_payload),
        expected_game_count=153,
    )

    with pytest.raises(DevelopmentPanelError, match="authority contract mismatch"):
        _development_panel._verify_authority(
            project_root=source_root,
            spec=spec,
        )


def test_verified_diagnostic_partition_iterator_loads_complete_bound_batch(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    spec = _source_fixture(source_root)
    publication = publish_exact153_development_panel(
        project_root=source_root,
        output_root=source_root / "published",
        source_spec=spec,
        confirmatory_evidence={},
    )

    partitions = list(
        _development_panel.iter_verified_diagnostic_panel_partitions(
            project_root=source_root,
            batch_manifest_path=publication.batch_manifest_path,
            batch_manifest_file_sha256=publication.batch_manifest_sha256,
            expected_game_count=1,
        )
    )

    assert len(partitions) == 1
    partition = partitions[0]
    assert isinstance(
        partition,
        _development_panel.VerifiedDiagnosticPanelPartition,
    )
    assert partition.game_id == GAME_ID
    assert partition.batch_game_count == 1
    assert partition.batch_manifest_file_sha256 == (
        publication.batch_manifest_sha256
    )
    assert set(partition.panel["game_id"]) == {GAME_ID}
    assert set(partition.panel["schema_version"]) == {
        "HistoricalTradesOnlyProbabilityPanelV2"
    }


def test_verified_diagnostic_partition_iterator_rejects_incomplete_batch(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    spec = _source_fixture(source_root)
    publication = publish_exact153_development_panel(
        project_root=source_root,
        output_root=source_root / "published",
        source_spec=spec,
        confirmatory_evidence={},
    )

    with pytest.raises(
        DevelopmentPanelError,
        match="complete diagnostic batch contract mismatch",
    ):
        list(
            _development_panel.iter_verified_diagnostic_panel_partitions(
                project_root=source_root,
                batch_manifest_path=publication.batch_manifest_path,
                batch_manifest_file_sha256=publication.batch_manifest_sha256,
                expected_game_count=2,
            )
        )


def test_verified_diagnostic_partition_iterator_checks_semantic_row_hash(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    spec = _source_fixture(source_root)
    publication = publish_exact153_development_panel(
        project_root=source_root,
        output_root=source_root / "published",
        source_spec=spec,
        confirmatory_evidence={},
    )
    output_root = publication.output_root
    batch = json.loads(publication.batch_manifest_path.read_text())
    game_descriptor = batch["games"][0]
    game_manifest_path = output_root / game_descriptor["manifest_path"]
    game_manifest = json.loads(game_manifest_path.read_text())
    diagnostic = next(
        descriptor
        for descriptor in game_manifest["tables"]
        if descriptor["name"] == "diagnostic_panel"
    )
    diagnostic["semantic_rows_sha256"] = "sha256:" + "f" * 64
    game_manifest.pop("bundle_sha256")
    game_manifest["bundle_sha256"] = (
        _development_panel._canonical_sha256(game_manifest)
    )
    game_payload = _canonical(game_manifest)
    game_file_sha = _sha(game_payload)
    game_hex = game_file_sha.removeprefix("sha256:")
    replacement_game_path = (
        output_root
        / "single-game"
        / GAME_ID
        / "manifests"
        / "sha256"
        / game_hex[:2]
        / f"{game_hex}.manifest.json"
    )
    replacement_game_path.parent.mkdir(parents=True, exist_ok=True)
    replacement_game_path.write_bytes(game_payload)
    game_descriptor.update(
        {
            "manifest_path": replacement_game_path.relative_to(
                output_root
            ).as_posix(),
            "manifest_sha256": game_file_sha,
            "bundle_sha256": game_manifest["bundle_sha256"],
        }
    )
    batch.pop("batch_sha256")
    batch["batch_sha256"] = _development_panel._canonical_sha256(batch)
    batch_payload = _canonical(batch)
    batch_file_sha = _sha(batch_payload)
    batch_hex = batch_file_sha.removeprefix("sha256:")
    replacement_batch_path = (
        output_root
        / "batches"
        / "manifests"
        / "sha256"
        / batch_hex[:2]
        / f"{batch_hex}.batch-index.json"
    )
    replacement_batch_path.parent.mkdir(parents=True, exist_ok=True)
    replacement_batch_path.write_bytes(batch_payload)

    with pytest.raises(
        DevelopmentPanelError,
        match="diagnostic_panel semantic rows mismatch",
    ):
        list(
            _development_panel.iter_verified_diagnostic_panel_partitions(
                project_root=source_root,
                batch_manifest_path=replacement_batch_path,
                batch_manifest_file_sha256=batch_file_sha,
                expected_game_count=1,
            )
        )


def test_runner_exposes_single_game_streaming_option() -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/research/run_nfl_x15_development_panel.py"),
            "--help",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--game-id" in result.stdout
    assert "--output-root" in result.stdout
