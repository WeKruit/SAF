from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prediction_market.research.nfl_x15_decision_context_v2 import (
    compute_prestate_context_sha256,
)
from prediction_market.research.nfl_x15_regulation_panel_v4 import (
    FEATURE_BLOCKS,
    FORMAL_MODEL_FORBIDDEN_COLUMNS,
    FORMAL_MODEL_REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    RegulationPanelV4Error,
    _atomic_publish,
    _canonical_sha256,
    _parquet_bytes,
    _parse_memory_free_percent,
    build_regulation_anchor_game_panel,
    publish_regulation_panel_v4_frames,
)
from prediction_market.research.nfl_x15_regulation_models import (
    load_verified_regulation_panel_batch,
    prepare_regulation_panel,
    verify_regulation_panel_batch_manifest,
)


_SHA = "sha256:" + "a" * 64


def _inputs(*, include_h_trade: bool = True) -> dict[str, object]:
    game_id = "2025_01_AWY_HME"
    anchor_specs = (
        ("anchor-td", "parent-score", "event-td", 10, 0.0, 1.0, True, "PASS"),
        ("anchor-try", "parent-score", "event-try", 20, 7.5, 8.0, True, "TRY"),
        (
            "anchor-review",
            "parent-review",
            "event-review",
            30,
            10.0,
            11.0,
            False,
            "ADMIN",
        ),
        (
            "anchor-pass",
            "parent-pass",
            "event-pass",
            40,
            20.0,
            21.0,
            True,
            "PASS",
        ),
    )
    base = pd.Timestamp("2025-09-01T00:00:00Z")
    anchors: list[dict[str, object]] = []
    contexts: list[dict[str, object]] = []
    for (
        anchor_id,
        parent_id,
        event_id,
        order,
        start_s,
        end_s,
        primary,
        action,
    ) in anchor_specs:
        start = base + pd.Timedelta(seconds=start_s)
        end = base + pd.Timedelta(seconds=end_s)
        supported = action != "ADMIN"
        tags = (
            ["TOUCHDOWN", "PASS_TOUCHDOWN"]
            if event_id == "event-td"
            else ["EXTRA_POINT_GOOD"]
            if event_id == "event-try"
            else ["REVIEW"]
            if event_id == "event-review"
            else ["PASS_ATTEMPT", "COMPLETE_PASS"]
        )
        anchors.append(
            {
                "schema_version": "ProvisionalInformationAnchorV2",
                "game_id": game_id,
                "information_anchor_id": anchor_id,
                "episode_id": parent_id,
                "event_id": event_id,
                "raw_play_id": str(order),
                "order_sequence": order,
                "source_interval_start": start,
                "source_interval_end": end,
                "information_anchor": end,
                "provisional_known_at": end if supported else pd.NaT,
                "provisional_primary_event_id": event_id,
                "provisional_primary_action": action,
                "provisional_outcome_tags": json.dumps(tags),
                "provisional_actor_team": "HME",
                "provisional_beneficiary_team": "HME",
                "provisional_score_points_observed": (
                    6 if event_id == "event-td" else 1 if action == "TRY" else 0
                ),
                "provisional_turnover_observed": False,
                "provisional_yards_gained": 12,
                "provisional_return_yards": 0,
                "provisional_feature_support_status": (
                    "SUPPORTED"
                    if supported
                    else "PRIMARY_FEATURE_SUPPORT_UNPROVEN"
                ),
                "provisional_feature_support_reason": (
                    "DIRECT_PROVISIONAL_SOURCE_ROW"
                    if supported
                    else "REVIEW_REVISION_CHRONOLOGY_UNAVAILABLE"
                ),
                "provisional_snapshot_sha256": _SHA,
                "canonical_stage_b_information_event_eligible": True,
                "primary_selection_eligible": primary,
                "censor_boundary_eligible": True,
                "pbp_source_sha256": _SHA,
            }
        )
        contexts.append(
            {
                "schema_version": "EventPrestateContextV2",
                "game_id": game_id,
                "information_anchor_id": anchor_id,
                "parent_episode_id_audit_only": parent_id,
                "event_id": event_id,
                "raw_play_id": str(order),
                "order_sequence": order,
                "source_interval_start": start,
                "source_interval_end": end,
                "information_anchor": end,
                "primary_selection_eligible": primary,
                "censor_boundary_eligible": True,
                "sports_prestate_known_at": end,
                "sports_prestate_known_at_basis": "FACT_KNOWN_AT",
                "quarter": 4,
                "is_overtime": False,
                "game_clock": "02:00",
                "game_seconds_remaining": 120,
                "home_team": "HME",
                "away_team": "AWY",
                "possession_team": "HME",
                "possession_is_home": True,
                "pre_home_score": 20,
                "pre_away_score": 20,
                "score_margin_home": 0,
                "down": 3,
                "distance": 7,
                "yardline_100": 35,
                "goal_to_go": False,
                "pre_red_zone": False,
                "home_timeouts_remaining": 2,
                "away_timeouts_remaining": 1,
                "p_before_home": 0.51,
                "p_before_home_missing": False,
                "prestate_support_status": "SUPPORTED",
                "pre_state_known_at": end - pd.Timedelta(milliseconds=100),
                "state_input_sha256": _SHA,
                "feature_provenance_row_count": 11,
                "feature_provenance_semantic_sha256": _SHA,
                "model_id": "MODEL",
                "model_version": "v1",
                "model_sha256": _SHA,
                "model_manifest_sha256": _SHA,
                "pbp_source_sha256": _SHA,
                "source_binding_sha256": _SHA,
                "prestate_context_sha256": _SHA,
            }
        )
    trade_rows = [
        {
            "trade_id": "trade-at-anchor",
            "game_id": game_id,
            "venue": "polymarket",
            "contract_id": "home-token",
            "source_time_utc": base + pd.Timedelta(seconds=1),
            "price": 0.49,
            "size": 1.0,
        },
        {
            "trade_id": "trade-a",
            "game_id": game_id,
            "venue": "polymarket",
            "contract_id": "home-token",
            "source_time_utc": base + pd.Timedelta(seconds=1.5),
            "price": 0.50,
            "size": 1.0,
        },
        {
            "trade_id": "trade-b",
            "game_id": game_id,
            "venue": "polymarket",
            "contract_id": "home-token",
            "source_time_utc": base + pd.Timedelta(seconds=2),
            "price": 0.52,
            "size": 1.0,
        },
        {
            "trade_id": "trade-c",
            "game_id": game_id,
            "venue": "polymarket",
            "contract_id": "home-token",
            "source_time_utc": base + pd.Timedelta(seconds=2),
            "price": 0.56,
            "size": 3.0,
        },
    ]
    if include_h_trade:
        trade_rows.append(
            {
                "trade_id": "trade-d",
                "game_id": game_id,
                "venue": "polymarket",
                "contract_id": "home-token",
                "source_time_utc": base + pd.Timedelta(seconds=4),
                "price": 0.565,
                "size": 2.0,
            }
        )
    context_frame = pd.DataFrame(contexts)
    context_frame["prestate_context_sha256"] = context_frame.apply(
        lambda row: compute_prestate_context_sha256(row.to_dict()),
        axis=1,
    )
    return {
        "information_anchors": pd.DataFrame(anchors),
        "context": context_frame,
        "market_rows": pd.DataFrame(trade_rows),
        "contracts": pd.DataFrame(
            [
                {
                    "game_id": game_id,
                    "venue": "polymarket",
                    "contract_id": "home-token",
                    "contract_role": "ACTUAL_HOME_OUTCOME",
                    "logical_market_id": (
                        "polymarket:2025_01_AWY_HME:winner"
                    ),
                    "market_family": "moneyline",
                    "market_period": "full_game",
                    "proposition_kind": "primitive",
                    "outcome_team": "HME",
                    "actual_home_contract_identity_sha256": _SHA,
                    "native_raw_manifest_sha256s_json": json.dumps(
                        [_SHA], separators=(",", ":")
                    ),
                    "native_raw_object_sha256s_json": json.dumps(
                        [_SHA], separators=(",", ":")
                    ),
                    "native_raw_manifest_set_sha256": _canonical_sha256(
                        [_SHA]
                    ),
                    "native_raw_object_set_sha256": _canonical_sha256(
                        [_SHA]
                    ),
                    "native_rule_evidence_sha256": pd.NA,
                    "native_rule_evidence_status": (
                        "RAW_METADATA_CAPTURED_RULE_SEMANTICS_NOT_ADJUDICATED"
                    ),
                    "realized_outcome_comparability_status": (
                        "UNPROVEN_NOT_ADJUDICATED"
                    ),
                    "realized_outcome_comparability_evidence_sha256": pd.NA,
                    "strict_contract_rule_equivalence_status": "UNPROVEN",
                    "strict_contract_rule_equivalence_evidence_sha256": pd.NA,
                    "market_continuity_start_utc": base,
                    "market_continuity_end_utc": (
                        base + pd.Timedelta(seconds=120)
                    ),
                    "market_continuity_support_status": (
                        "VERIFIED_CAPTURE_PAGINATION"
                    ),
                    "market_continuity_evidence_sha256": _SHA,
                }
            ]
        ),
        "cohort_row": {
            "game_id": game_id,
            "nfl_week": 1,
            "authority_sha256": _SHA,
        },
        "source_hashes": {
            "information_anchor_manifest_sha256": _SHA,
            "information_anchor_object_sha256": _SHA,
            "context_manifest_sha256": _SHA,
            "context_object_sha256": _SHA,
            "market_game_manifest_sha256": _SHA,
            "market_observations_object_sha256": _SHA,
            "market_inventory_object_sha256": _SHA,
            "market_capture_batch_sha256": _SHA,
            "market_capture_index_sha256": _SHA,
            "market_capture_checkpoint_sha256": _SHA,
            "cohort_authority_sha256": _SHA,
        },
    }


def test_panel_v4_uses_pia_grain_and_actual_trade_bucket() -> None:
    result = build_regulation_anchor_game_panel(
        **_inputs(),
        landmark_seconds=(1,),
        endpoint_seconds=(5, 10),
    )
    row = result.panel.loc[
        result.panel["information_anchor_id"].eq("anchor-td")
        & result.panel["endpoint_seconds"].eq(5)
    ].iloc[0]

    assert row["schema_version"] == SCHEMA_VERSION
    assert row["anchor_kind"] == "PROVISIONAL_FIRST_SEEN"
    assert row["analysis_role"] == "PRIMARY_SELECTION"
    assert row["primary_feature_snapshot_sha256"] == _SHA
    assert row["prestate_context_sha256"].startswith("sha256:")
    assert row["mark_l_price"] == pytest.approx(0.55)
    assert row["mark_l_observation_count"] == 2
    assert row["mark_l_trade_ids_json"] == '["trade-b","trade-c"]'
    assert row["availability_h"] == 1
    assert row["mark_h_price"] == pytest.approx(0.565)
    assert row["direction_h"] == "UP"
    assert row["loss_eligible"]
    assert not row["censored"]
    assert row["next_information_anchor_id"] == "anchor-try"
    assert (
        row["next_censor_source_interval_start"]
        == pd.Timestamp("2025-09-01T00:00:07.500Z")
    )


def test_same_parent_try_censors_td_window_and_is_not_no_trade() -> None:
    result = build_regulation_anchor_game_panel(
        **_inputs(),
        landmark_seconds=(1,),
        endpoint_seconds=(5, 10),
    )
    row = result.panel.loc[
        result.panel["information_anchor_id"].eq("anchor-td")
        & result.panel["endpoint_seconds"].eq(10)
    ].iloc[0]

    assert row["episode_id"] == "parent-score"
    assert row["next_parent_episode_id_audit_only"] == "parent-score"
    assert row["censored"]
    assert row["censor_reason"] == "NEXT_INFORMATION_INTERVAL_IN_L_H_WINDOW"
    assert not row["loss_eligible"]
    assert pd.isna(row["availability_h"])
    assert pd.isna(row["direction_h"])


def test_censor_only_review_still_terminates_try_window() -> None:
    result = build_regulation_anchor_game_panel(
        **_inputs(),
        landmark_seconds=(1,),
        endpoint_seconds=(5,),
    )
    row = result.panel.loc[
        result.panel["information_anchor_id"].eq("anchor-try")
    ].iloc[0]

    assert row["next_information_anchor_id"] == "anchor-review"
    assert row["censored"]
    assert row["censor_reason"] == "NEXT_INFORMATION_INTERVAL_IN_L_H_WINDOW"
    assert "anchor-review" not in set(
        result.panel["information_anchor_id"].astype(str)
    )


def test_overlapping_next_interval_is_order_ambiguous_and_excluded() -> None:
    inputs = _inputs()
    anchors = inputs["information_anchors"]
    context = inputs["context"]
    assert isinstance(anchors, pd.DataFrame)
    assert isinstance(context, pd.DataFrame)
    anchors.loc[
        anchors["information_anchor_id"].eq("anchor-try"),
        "source_interval_start",
    ] = pd.Timestamp("2025-09-01T00:00:00.500Z")
    context.loc[
        context["information_anchor_id"].eq("anchor-try"),
        "source_interval_start",
    ] = pd.Timestamp("2025-09-01T00:00:00.500Z")

    result = build_regulation_anchor_game_panel(
        **inputs,
        landmark_seconds=(1,),
        endpoint_seconds=(5,),
    )
    row = result.panel.loc[
        result.panel["information_anchor_id"].eq("anchor-td")
    ].iloc[0]

    assert row["order_ambiguous"]
    assert row["censored"]
    assert row["censor_reason"] == "NEXT_INFORMATION_INTERVAL_OVERLAPS_FOCAL"
    assert row["landmark_eligible"]
    assert row["continuity_valid"]
    assert not row["loss_eligible"]
    assert pd.isna(row["availability_h"])


def test_clean_window_without_new_trade_is_availability_zero() -> None:
    result = build_regulation_anchor_game_panel(
        **_inputs(include_h_trade=False),
        landmark_seconds=(1,),
        endpoint_seconds=(5,),
    )
    row = result.panel.loc[
        result.panel["information_anchor_id"].eq("anchor-td")
    ].iloc[0]

    assert row["landmark_eligible"]
    assert row["continuity_valid"]
    assert row["loss_eligible"]
    assert row["availability_h"] == 0
    assert pd.isna(row["direction_h"])
    assert row["mark_h_status"] == "NO_ACTUAL_TRADE"


def test_context_future_or_final_columns_fail_closed() -> None:
    inputs = _inputs()
    context = inputs["context"]
    assert isinstance(context, pd.DataFrame)
    context["final_score_points"] = 0
    with pytest.raises(RegulationPanelV4Error, match="forbidden"):
        build_regulation_anchor_game_panel(
            **inputs,
            landmark_seconds=(1,),
            endpoint_seconds=(5,),
        )


def test_unknown_transport_audit_status_fails_closed() -> None:
    inputs = _inputs()
    contracts = inputs["contracts"]
    assert isinstance(contracts, pd.DataFrame)
    contracts.loc[
        :, "realized_outcome_comparability_status"
    ] = "SUPPORTED"
    with pytest.raises(RegulationPanelV4Error, match="status/hash mismatch"):
        build_regulation_anchor_game_panel(
            **inputs,
            landmark_seconds=(1,),
            endpoint_seconds=(5,),
        )


def test_feature_contract_has_no_finalized_or_duplicate_factor_columns() -> None:
    result = build_regulation_anchor_game_panel(
        **_inputs(),
        landmark_seconds=(1,),
        endpoint_seconds=(5,),
    )

    assert set(FEATURE_BLOCKS["D3"]).issubset(result.panel.columns)
    assert set(FORMAL_MODEL_REQUIRED_COLUMNS).issubset(result.panel.columns)
    assert not FORMAL_MODEL_FORBIDDEN_COLUMNS.intersection(
        result.panel.columns
    )
    assert not any(
        column.startswith(("final_", "post_", "factor__", "event_tag__"))
        for column in result.panel.columns
    )
    assert result.panel["source_row_id"].str.fullmatch(
        r"sha256:[0-9a-f]{64}"
    ).all()
    assert result.panel[
        ["information_anchor_id", "venue", "landmark_seconds", "endpoint_seconds"]
    ].duplicated().sum() == 0


def test_missing_p_before_has_null_model_known_at_and_explicit_status() -> None:
    inputs = _inputs()
    context = inputs["context"]
    assert isinstance(context, pd.DataFrame)
    selected = context["information_anchor_id"].eq("anchor-td")
    context.loc[selected, "p_before_home"] = pd.NA
    context.loc[selected, "p_before_home_missing"] = True
    context.loc[selected, "prestate_support_status"] = "MISSING_PRE_STATE"
    context.loc[selected, "prestate_context_sha256"] = context.loc[
        selected
    ].apply(
        lambda row: compute_prestate_context_sha256(row.to_dict()),
        axis=1,
    )

    row = build_regulation_anchor_game_panel(
        **inputs,
        landmark_seconds=(1,),
        endpoint_seconds=(5,),
    ).panel.loc[lambda frame: frame["information_anchor_id"].eq("anchor-td")].iloc[
        0
    ]

    assert row["p_before_support_status"] == "MISSING"
    assert pd.isna(row["prestate_known_at"])
    assert row["p_before_home_missing"]


def test_memory_pressure_parser_reads_real_format() -> None:
    assert (
        _parse_memory_free_percent(
            "System-wide memory free percentage: 63%\n"
        )
        == 63
    )
    assert _parse_memory_free_percent("not a memory report") is None


def test_atomic_publish_never_overwrites_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "objects" / "object.bin"
    original_link = __import__("os").link

    def racing_link(source: object, destination: object) -> None:
        Path(destination).write_bytes(b"other-bytes")
        raise FileExistsError

    monkeypatch.setattr("os.link", racing_link)
    with pytest.raises(RegulationPanelV4Error, match="collision"):
        _atomic_publish(target, b"expected-bytes")
    assert target.read_bytes() == b"other-bytes"
    monkeypatch.setattr("os.link", original_link)


def test_parquet_fingerprint_uses_exact_persisted_schema() -> None:
    frame = pd.DataFrame(
        {
            "all_missing_utc": pd.to_datetime([None, None], utc=True),
            "value": pd.Series([1, 2], dtype="Int64"),
        }
    )

    payload, schema_fingerprint = _parquet_bytes(frame)
    persisted_schema = pq.read_schema(
        pa.BufferReader(payload)
    ).remove_metadata()
    expected = "sha256:" + hashlib.sha256(
        persisted_schema.serialize().to_pybytes()
    ).hexdigest()

    assert schema_fingerprint == expected


def test_published_panel_verifies_loads_and_prepares_for_model(
    tmp_path: Path,
) -> None:
    inputs = _inputs()
    built = build_regulation_anchor_game_panel(
        **inputs,
        landmark_seconds=(1, 2, 3, 5, 10),
        endpoint_seconds=tuple(range(5, 61, 5)),
    )
    game_id = "2025_01_AWY_HME"
    published = publish_regulation_panel_v4_frames(
        project_root=tmp_path,
        output_root=Path("artifacts/panel-v4-fixture"),
        game_panels={game_id: built},
        game_source_hashes={
            game_id: inputs["source_hashes"],  # type: ignore[dict-item]
        },
        batch_sources={
            "market_batch_file_sha256": _SHA,
            "market_capture_batch_sha256": _SHA,
            "information_anchor_manifest_file_sha256": _SHA,
            "information_anchor_object_sha256": _SHA,
            "context_manifest_file_sha256": _SHA,
            "context_object_sha256": _SHA,
            "cohort_authority_sha256": _SHA,
            "cohort_mapping_sha256": _SHA,
        },
        expected_game_count=1,
    )

    manifest = json.loads(published.batch_manifest_path.read_text())
    game_manifest_path = (
        published.output_root / manifest["games"][0]["manifest_path"]
    )
    game_manifest = json.loads(game_manifest_path.read_text())
    panel_descriptor = next(
        table
        for table in game_manifest["tables"]
        if table["name"] == "regulation_decision_panel"
    )
    panel_path = published.output_root / panel_descriptor["object_path"]
    persisted_schema = pq.read_schema(panel_path).remove_metadata()
    persisted_schema_fingerprint = "sha256:" + hashlib.sha256(
        persisted_schema.serialize().to_pybytes()
    ).hexdigest()

    verified = verify_regulation_panel_batch_manifest(
        project_root=tmp_path,
        batch_manifest_path=published.batch_manifest_path,
        batch_manifest_file_sha256=published.batch_manifest_sha256,
        expected_game_count=1,
    )
    loaded = load_verified_regulation_panel_batch(
        project_root=tmp_path,
        batch_manifest_path=published.batch_manifest_path,
        batch_manifest_file_sha256=published.batch_manifest_sha256,
        venue_scope=("POLYMARKET",),
        expected_game_count=1,
    )
    prepared = prepare_regulation_panel(loaded.frame)

    assert verified.game_count == 1
    assert (
        panel_descriptor["schema_fingerprint"]
        == persisted_schema_fingerprint
    )
    assert loaded.source_row_count == published.panel_row_count
    assert len(prepared) == published.loss_eligible_count
