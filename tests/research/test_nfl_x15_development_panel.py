from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

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


def _publish_parquet(root: Path, relative_prefix: str, frame: pd.DataFrame) -> dict:
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
        "object_path": relative.relative_to(relative_prefix).as_posix(),
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
) -> dict:
    descriptors = {
        name: _publish_parquet(root, relative_prefix, frame)
        for name, frame in tables.items()
    }
    material: dict[str, object] = {
        "schema": schema,
        "game_id": game_id,
        "cohort": "development",
        "publication_gate": "PASS",
        "holdout_reaction_accessed": False,
    }
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
        "manifest_path": relative.relative_to(relative_prefix).as_posix(),
        "manifest_sha256": digest,
        "bundle_sha256": material["bundle_sha256"],
    }


def _publish_batch(
    root: Path,
    relative_prefix: str,
    *,
    schema: str,
    game: dict,
    market_style: bool,
) -> tuple[Path, str]:
    material: dict[str, object] = {
        "schema": schema,
        "experiment_id": "X-13" if "stage_a" not in schema else "X-15",
        "cohort": "development",
        "game_count": 1,
        "games": [game],
        "publication_gate": "PASS",
    }
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


def _source_fixture(tmp_path: Path) -> DevelopmentSourceSpec:
    facts, hits = _facts()
    observations, inventory = _market()
    facts_game = _publish_game_manifest(
        tmp_path,
        "facts",
        game_id=GAME_ID,
        schema="nfl_x13_exact153_fact_single_game_manifest_v1",
        tables={"canonical_factor_events": facts, "factor_hits": hits},
        market_style=False,
    )
    refs_game = _publish_game_manifest(
        tmp_path,
        "stage-a",
        game_id=GAME_ID,
        schema="nfl_x15_stage_a_single_game_manifest_v1",
        tables={"reference_observations": _references()},
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
        "facts",
        schema="nfl_x13_exact153_fact_batch_index_v1",
        game=facts_game,
        market_style=False,
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


def _confirmatory_evidence(venue: str) -> VenueConfirmatoryEvidence:
    return VenueConfirmatoryEvidence(
        venue=venue,
        tick_rule_id=f"{venue}-historical-rule",
        tick_rules=pd.DataFrame(
            [
                {
                    "venue": venue,
                    "tick_rule_id": f"{venue}-historical-rule",
                    "effective_start_utc": "2025-09-07T00:00:00Z",
                    "effective_end_utc": "2025-09-08T00:00:00Z",
                    "tick_size": 0.01,
                }
            ]
        ),
        market_continuity=pd.DataFrame(
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
        ),
        tick_rule_source_sha256="sha256:" + ("1" if venue == "polymarket" else "2") * 64,
        continuity_source_sha256="sha256:" + ("3" if venue == "polymarket" else "4") * 64,
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
        "HistoricalTradesOnlyProbabilityPanelV1"
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
