from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from prediction_market.research.nfl_x15_native_rule_audit import (
    NativeRuleInput,
    RuleAuditError,
    RuleAuditFrames,
    build_game_rule_audit,
    canonical_team_code,
    derive_final_game_facts,
    extract_kalshi_rule_input,
    extract_polymarket_rule_input,
    filter_development_capture_entries,
    kalshi_team_code,
    publish_rule_audit_frames,
    select_metadata_raw_refs,
    verify_rule_audit_batch,
)


_SHA = "sha256:" + "a" * 64


def _native(
    venue: str,
    *,
    rule_texts: tuple[str, ...],
    resolution_source: str | None,
) -> NativeRuleInput:
    suffix = "poly" if venue == "POLYMARKET" else "kalshi"
    return NativeRuleInput(
        venue=venue,
        contract_ids=(
            ("poly-market",)
            if venue == "POLYMARKET"
            else ("KXNFLGAME-25SEP07JACKC-JAC", "KXNFLGAME-25SEP07JACKC-KC")
        ),
        identity_material={
            "logical_market": f"{suffix}-winner",
            "home_team": "JAX",
            "away_team": "KC",
        },
        rule_texts=rule_texts,
        resolution_source=resolution_source,
        raw_manifest_sha256s=(_SHA,),
        raw_object_sha256s=(_SHA,),
        request_sha256s=(_SHA,),
    )


def _fixture_frames() -> RuleAuditFrames:
    polymarket = _native(
        "POLYMARKET",
        rule_texts=(
            "If the Jacksonville Jaguars win, the market resolves to Jaguars. "
            "If the Kansas City Chiefs win, the market resolves to Chiefs. "
            "If postponed, the market remains open until completed. "
            "If canceled with no make-up game, the market resolves 50-50.",
        ),
        resolution_source="https://www.nfl.com/",
    )
    kalshi = _native(
        "KALSHI",
        rule_texts=(
            "If Jacksonville wins the game, the market resolves to Yes.",
            "If Kansas City wins the game, the market resolves to Yes.",
            "If postponed or delayed, the market remains open until the "
            "rescheduled game finishes. If cancelled and not played, "
            "rescheduled, or declared a tie, all teams resolve 50/50.",
        ),
        resolution_source=None,
    )
    evidence, pair = build_game_rule_audit(
        game_id="2025_05_KC_JAX",
        home_team="JAX",
        away_team="KC",
        final_home_score=31,
        final_away_score=28,
        facts_manifest_sha256=_SHA,
        facts_object_sha256=_SHA,
        polymarket=polymarket,
        kalshi=kalshi,
    )
    return RuleAuditFrames(
        venue_evidence=pd.DataFrame(evidence),
        contract_pairs=pd.DataFrame([pair]),
        read_audit={
            "metadata_object_reads": 2,
            "trade_object_reads": 0,
            "candle_object_reads": 0,
            "reaction_object_reads": 0,
            "holdout_game_index_reads": 0,
            "holdout_fact_object_reads": 0,
        },
    )


def test_jax_alias_is_explicit_and_round_trips() -> None:
    assert kalshi_team_code("JAX") == "JAC"
    assert canonical_team_code("JAC", venue="KALSHI") == "JAX"
    assert kalshi_team_code("KC") == "KC"


def test_completed_non_tie_is_comparable_but_clause_mismatch_is_not_equivalent() -> None:
    frames = _fixture_frames()
    pair = frames.contract_pairs.iloc[0]

    assert pair["completed_non_tie_comparability_status"] == "COMPARABLE"
    assert pair["strict_rule_equivalence_status"] == "NON_EQUIVALENT"
    assert "tie_clause" in json.loads(pair["strict_mismatch_clauses_json"])


def test_cohort_filter_never_returns_holdout_game_indexes() -> None:
    entries = [
        {"game_id": "2025_01_AAA_BBB", "index_path": "development.json"},
        {"game_id": "2025_13_AAA_BBB", "index_path": "holdout.json"},
    ]

    selected = filter_development_capture_entries(
        entries=entries,
        development_game_ids=("2025_01_AAA_BBB",),
        holdout_game_ids=("2025_13_AAA_BBB",),
    )

    assert [row["game_id"] for row in selected] == ["2025_01_AAA_BBB"]


def test_metadata_selector_never_opens_trade_or_reaction_objects() -> None:
    refs = [
        {"manifest_sha256": "poly-meta", "object_path": "poly-meta.json"},
        {"manifest_sha256": "kalshi-meta", "object_path": "kalshi-meta.json"},
        {"manifest_sha256": "trade", "object_path": "trade.json"},
        {"manifest_sha256": "reaction", "object_path": "reaction.parquet"},
    ]
    manifests = {
        "poly-meta": {
            "source_url": "https://gamma-api.polymarket.com/events/slug/nfl-a-b"
        },
        "kalshi-meta": {
            "source_url": (
                "https://external-api.kalshi.com/trade-api/v2/"
                "historical/markets"
            )
        },
        "trade": {"source_url": "https://data-api.polymarket.com/trades"},
        "reaction": {"source_url": "internal://reaction_paths"},
    }

    selected = select_metadata_raw_refs(
        raw_refs=refs,
        manifest_loader=lambda row: manifests[str(row["manifest_sha256"])],
    )

    assert [row["object_path"] for row in selected] == [
        "poly-meta.json",
        "kalshi-meta.json",
    ]


def test_native_metadata_extracts_exact_contracts_and_jax_alias() -> None:
    lineage = {
        "manifest_sha256": _SHA,
        "object_sha256": _SHA,
        "request_sha256": _SHA,
    }
    polymarket = extract_polymarket_rule_input(
        game_id="2025_05_KC_JAX",
        home_team="JAX",
        away_team="KC",
        identity={
            "polymarket_market_id": "123",
            "polymarket_condition_id": "0xabc",
        },
        payload={
            "teams": [
                {
                    "ordering": "away",
                    "abbreviation": "kc",
                    "alias": "Chiefs",
                },
                {
                    "ordering": "home",
                    "abbreviation": "jax",
                    "alias": "Jaguars",
                },
            ],
            "markets": [
                {
                    "id": "123",
                    "conditionId": "0xabc",
                    "question": "Chiefs vs. Jaguars",
                    "outcomes": '["Chiefs","Jaguars"]',
                    "clobTokenIds": '["token-kc","token-jax"]',
                    "description": (
                        "If the Kansas City Chiefs win, resolve Chiefs. "
                        "If the Jacksonville Jaguars win, resolve Jaguars. "
                        "If postponed, remain open until completed. "
                        "If canceled with no make-up, resolve 50-50."
                    ),
                    "resolutionSource": "https://www.nfl.com/",
                }
            ],
        },
        lineage=lineage,
    )
    kalshi = extract_kalshi_rule_input(
        game_id="2025_05_KC_JAX",
        home_team="JAX",
        away_team="KC",
        identity={
            "kalshi_event_ticker": "KXNFLGAME-25OCT05KCJAC",
            "kalshi_team_tickers": [
                "KXNFLGAME-25OCT05KCJAC-KC",
                "KXNFLGAME-25OCT05KCJAC-JAC",
            ],
        },
        payload={
            "markets": [
                {
                    "ticker": "KXNFLGAME-25OCT05KCJAC-KC",
                    "event_ticker": "KXNFLGAME-25OCT05KCJAC",
                    "rules_primary": (
                        "If Kansas City wins the game, resolves to Yes."
                    ),
                    "rules_secondary": (
                        "If postponed, remain open until rescheduled. "
                        "If cancelled or tied, resolve 50/50."
                    ),
                },
                {
                    "ticker": "KXNFLGAME-25OCT05KCJAC-JAC",
                    "event_ticker": "KXNFLGAME-25OCT05KCJAC",
                    "rules_primary": (
                        "If Jacksonville wins the game, resolves to Yes."
                    ),
                    "rules_secondary": (
                        "If postponed, remain open until rescheduled. "
                        "If cancelled or tied, resolve 50/50."
                    ),
                },
            ]
        },
        lineage=lineage,
    )

    assert polymarket.contract_ids == ("123",)
    assert kalshi.contract_ids == (
        "KXNFLGAME-25OCT05KCJAC-JAC",
        "KXNFLGAME-25OCT05KCJAC-KC",
    )
    assert kalshi.identity_material["native_home_team"] == "JAC"
    assert kalshi.identity_material["home_team"] == "JAX"


def test_final_facts_uses_last_canonical_state_and_rejects_tie() -> None:
    frame = pd.DataFrame(
        [
            {
                "game_id": "2025_01_AAA_BBB",
                "order_sequence": 1,
                "home_team": "BBB",
                "away_team": "AAA",
                "post_home_score": 7,
                "post_away_score": 0,
            },
            {
                "game_id": "2025_01_AAA_BBB",
                "order_sequence": 2,
                "home_team": "BBB",
                "away_team": "AAA",
                "post_home_score": 21,
                "post_away_score": 17,
            },
        ]
    )
    facts = derive_final_game_facts(
        game_id="2025_01_AAA_BBB",
        events=frame,
        facts_manifest_sha256=_SHA,
        facts_object_sha256=_SHA,
    )
    assert facts["final_home_score"] == 21
    assert facts["final_away_score"] == 17
    assert facts["completed_non_tie"] is True


def test_rule_audit_cli_supports_validate_only() -> None:
    project = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(
                project
                / "scripts"
                / "research"
                / "build_nfl_x15_native_rule_audit.py"
            ),
            "--help",
        ],
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--validate-only" in completed.stdout


def test_published_audit_verifies_and_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    published = publish_rule_audit_frames(
        project_root=tmp_path,
        output_root=Path("artifacts/native-rule-audit-v1"),
        frames=_fixture_frames(),
        source_bindings={
            "governance_object_sha256": _SHA,
            "capture_batch_file_sha256": _SHA,
            "facts_batch_file_sha256": _SHA,
        },
        expected_game_count=1,
    )
    verified = verify_rule_audit_batch(
        project_root=tmp_path,
        manifest_path=published.manifest_path,
        manifest_file_sha256=published.manifest_file_sha256,
        expected_game_count=1,
    )
    assert verified.venue_evidence_row_count == 2
    assert verified.contract_pair_row_count == 1

    manifest = json.loads(published.manifest_path.read_text())
    evidence = next(
        table
        for table in manifest["tables"]
        if table["name"] == "venue_native_rule_evidence"
    )
    object_path = published.output_root / evidence["object_path"]
    object_path.write_bytes(object_path.read_bytes() + b"tamper")
    with pytest.raises(RuleAuditError, match="hash mismatch"):
        verify_rule_audit_batch(
            project_root=tmp_path,
            manifest_path=published.manifest_path,
            manifest_file_sha256=published.manifest_file_sha256,
            expected_game_count=1,
        )
