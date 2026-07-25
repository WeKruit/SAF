from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from prediction_market.sports import nfl_x13_signal_exploration as exploration


def _frames(
    *,
    games: int = 6,
    audited_orientation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episodes: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    superclasses = ("score", "turnover", "routine")
    for game_index in range(games):
        game_id = f"2025_{game_index + 1:02d}_AAA_BBB"
        for event_index, superclass in enumerate(superclasses):
            episode_id = f"{game_id}:episode:{event_index}"
            episodes.append(
                {
                    "episode_id": episode_id,
                    "game_id": game_id,
                    "episode_superclass": superclass,
                }
            )
            for venue, contract_id, base in (
                ("kalshi", "kalshi:AAA", 0.02),
                ("polymarket", "poly:AAA", 0.018),
            ):
                for delay_seconds in (0, 5):
                    for horizon_seconds in (10, 60):
                        signed = (
                            base
                            + 0.002 * game_index
                            + 0.001 * event_index
                            - 0.0001 * delay_seconds
                        )
                        observations.append(
                            {
                                "game_id": game_id,
                                "episode_id": episode_id,
                                "episode_type": (
                                    "touchdown"
                                    if superclass == "score"
                                    else "pass"
                                    if superclass == "turnover"
                                    else "punt"
                                ),
                                "source_time_start_utc": (
                                    f"2025-09-{game_index + 1:02d}"
                                    f"T00:00:0{event_index + 1}Z"
                                ),
                                "source_time_end_utc": (
                                    f"2025-09-{game_index + 1:02d}"
                                    f"T00:00:0{event_index + 2}Z"
                                ),
                                "contract_id": contract_id,
                                "outcome": "AAA",
                                "signal_focal_team": "AAA",
                                "scoring_team": "AAA",
                                "pre_offense_team": "AAA",
                                "pre_defense_team": "AAA",
                                "pre_possession_team": "AAA",
                                "contract_measure": "winner",
                                "pre_game_seconds_remaining": 1000,
                                "pre_quarter": 3,
                                "pre_score_margin_home": 3,
                                "pre_down": 3,
                                "pre_distance": 7,
                                "pre_yardline_100": 15,
                                "pre_goal_to_go": False,
                                "venue": venue,
                                "market_family": "moneyline",
                                "delay_seconds": delay_seconds,
                                "horizon_seconds": horizon_seconds,
                                "signed_price_change": signed,
                                "pre_event_actual_trade": 0.5,
                                "first_post_event_trade": 0.5 + signed,
                                "vwap": 0.5 + signed / 2,
                                "maximum_excursion": abs(signed) + 0.01,
                                "net_change_60s": signed + 0.002,
                                "overshoot": bool(game_index % 2),
                                "reversal": False,
                                "trade_count": 2 + game_index,
                                "volume": 10.0 + game_index,
                                "staleness_seconds": float(game_index),
                                "signal_event_eligible": True,
                                "signal_candidate_eligible": True,
                                "pre_trade_fresh_le_60s": True,
                                "pre_trade_age_seconds": 1.0,
                                "pre_trade_age_status": (
                                    "RECONSTRUCTED_FROM_FROZEN_NORMALIZED_OBSERVATIONS"
                                ),
                                "validity_status": "OBSERVED",
                                "order_ambiguous": False,
                                "contaminated": False,
                            }
                        )
    orientations = pd.DataFrame(
        [
            {
                "contract_id": "kalshi:AAA",
                "outcome": "AAA",
                "semantic_orientation_id": "winner:AAA",
                "orientation_audited": audited_orientation,
                "event_relative_alignment": "event_team",
            },
            {
                "contract_id": "poly:AAA",
                "outcome": "AAA",
                "semantic_orientation_id": "winner:AAA",
                "orientation_audited": audited_orientation,
                "event_relative_alignment": "event_team",
            },
        ]
    )
    return (
        pd.DataFrame(observations),
        pd.DataFrame(episodes),
        orientations,
    )


def _config() -> exploration.SignalExplorationConfig:
    return exploration.SignalExplorationConfig(
        registered_delays_seconds=(0, 5),
        registered_horizons_seconds=(10, 60),
        bootstrap_samples=10_000,
        confidence_level=0.95,
        seed=20260725,
        minimum_episode_count=4,
        minimum_game_count=3,
    )


@pytest.fixture(autouse=True)
def _synthetic_frozen_games(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        exploration,
        "X13_GAME_IDS",
        tuple(f"2025_{index + 1:02d}_AAA_BBB" for index in range(6)),
    )


def test_exploration_preserves_sensitivity_and_reports_required_descriptives() -> None:
    observed, episodes, orientations = _frames()

    result = exploration.explore_signal(
        observed_cache=observed,
        episode_dim=episodes,
        contract_orientation_dim=orientations,
        config=_config(),
    )

    assert result["result_label"] == "PRELIMINARY_SOURCE_TIME_ONLY"
    assert result["game_count"] == 6
    assert set(result["coverage"].columns) == {
        "episode_superclass",
        "venue",
        "family",
        "delay_scenario_seconds",
        "horizon_seconds",
        "episode_count",
        "game_count",
        "observation_count",
    }
    assert len(result["coverage"]) == 3 * 2 * 1 * 2 * 2
    assert set(result["coverage"]["delay_scenario_seconds"]) == {0, 5}
    assert set(result["coverage"]["horizon_seconds"]) == {10, 60}

    summary = result["reaction_summary"]
    assert set(summary["episode_superclass"]) == {"score", "turnover", "routine"}
    required = {
        "signed_price_change_mean",
        "maximum_excursion_mean",
        "net_change_60s_mean",
        "overshoot_rate",
        "reversal_rate",
        "trade_count_sum",
        "volume_sum",
        "staleness_seconds_mean",
        "game_cluster_ci_low",
        "game_cluster_ci_high",
        "loo_sign_stable",
        "loo_sign_stability_rate",
        "maximum_single_game_contribution",
        "loo_estimates",
        "raw_p_value",
        "bh_q_value",
        "candidate_label",
    }
    assert required <= set(summary.columns)
    primary = (summary["delay_scenario_seconds"] == 0) & (
        summary["horizon_seconds"] == 10
    )
    assert (summary["candidate_label"] == "DESCRIPTIVE").all()
    assert not summary.loc[primary, "delay_horizon_sign_stable"].any()
    assert (summary["game_cluster_ci_low"] > 0).all()
    assert summary["loo_sign_stable"].all()
    assert (summary["bh_q_value"] >= summary["raw_p_value"]).all()
    assert result["claim_boundary"] == {
        "causal": False,
        "latency": False,
        "execution": False,
        "tradeability": False,
        "alpha": False,
        "order_flow_imbalance": False,
        "depth": False,
    }
    assert len(result["primary_contrasts"]) == 3
    assert len(result["path_comparisons"]) == 3
    assert len(result["state_moderators"]) == 15
    assert len(result["venue_comparisons"]) == 3
    assert all(
        {"raw_p_value", "bh_q_value", "candidate_label"} <= set(result[name])
        for name in (
            "primary_contrasts",
            "path_comparisons",
            "state_moderators",
            "venue_comparisons",
        )
    )


def test_all_reaction_gates_can_produce_signal_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed, episodes, orientations = _frames(games=20)
    seed_rows = observed[
        observed["delay_seconds"].eq(0) & observed["horizon_seconds"].eq(10)
    ]
    expanded = []
    for delay in (0, 1, 2):
        for horizon in (5, 10, 30):
            frame = seed_rows.copy()
            frame["delay_seconds"] = delay
            frame["horizon_seconds"] = horizon
            expanded.append(frame)
    observed = pd.concat(expanded, ignore_index=True)
    monkeypatch.setattr(
        exploration,
        "X13_GAME_IDS",
        tuple(sorted(episodes["game_id"].unique())),
    )
    config = exploration.SignalExplorationConfig(
        registered_delays_seconds=(0, 1, 2),
        registered_horizons_seconds=(5, 10, 30),
        bootstrap_samples=10_000,
        minimum_episode_count=20,
        minimum_game_count=6,
    )

    result = exploration.explore_signal(
        observed_cache=observed,
        episode_dim=episodes,
        contract_orientation_dim=orientations,
        config=config,
    )

    primary = result["reaction_summary"][
        result["reaction_summary"]["delay_scenario_seconds"].eq(0)
        & result["reaction_summary"]["horizon_seconds"].eq(10)
    ]
    assert set(primary["candidate_label"]) == {"SIGNAL_CANDIDATE"}


def test_cross_venue_symmetry_requires_audited_semantic_orientation() -> None:
    observed, episodes, orientations = _frames(audited_orientation=False)

    with pytest.raises(
        exploration.SignalExplorationError,
        match="no rows satisfy the frozen signal candidate analysis gate",
    ):
        exploration.explore_signal(
            observed_cache=observed,
            episode_dim=episodes,
            contract_orientation_dim=orientations,
            config=_config(),
        )


def test_cross_venue_symmetry_keeps_episode_as_shared_sports_unit() -> None:
    observed, episodes, orientations = _frames()

    result = exploration.explore_signal(
        observed_cache=observed,
        episode_dim=episodes,
        contract_orientation_dim=orientations,
        config=_config(),
    )

    symmetry = result["cross_venue_symmetry"]
    assert not symmetry.empty
    assert (symmetry["shared_unit"] == "episode_id").all()
    assert (symmetry["venue_count"] == 2).all()
    assert (symmetry["shared_episode_count"] == 6).all()
    assert (symmetry["shared_game_count"] == 6).all()


def test_contract_orientation_is_keyed_by_contract_and_outcome() -> None:
    observed, episodes, orientations = _frames()
    unused_other_outcomes = orientations.assign(
        outcome="BBB",
        semantic_orientation_id="winner:BBB",
        orientation_audited=False,
    )
    orientations = pd.concat(
        [orientations, unused_other_outcomes],
        ignore_index=True,
    )

    result = exploration.explore_signal(
        observed_cache=observed,
        episode_dim=episodes,
        contract_orientation_dim=orientations,
        config=_config(),
    )

    assert not result["cross_venue_symmetry"].empty
    assert result["reaction_summary"]["orientation_available"].all()


def test_orientation_audit_rejects_integer_truth_value() -> None:
    observed, episodes, orientations = _frames()
    orientations["orientation_audited"] = 1

    with pytest.raises(
        exploration.SignalExplorationError,
        match="orientation_audited must contain booleans",
    ):
        exploration.explore_signal(
            observed_cache=observed,
            episode_dim=episodes,
            contract_orientation_dim=orientations,
            config=_config(),
        )


def test_candidate_thresholds_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed, episodes, orientations = _frames(games=1)
    monkeypatch.setattr(
        exploration,
        "X13_GAME_IDS",
        ("2025_01_AAA_BBB",),
    )
    config = exploration.SignalExplorationConfig(
        registered_delays_seconds=(0, 5),
        registered_horizons_seconds=(10, 60),
        bootstrap_samples=10_000,
        confidence_level=0.95,
        seed=20260725,
        minimum_episode_count=4,
        minimum_game_count=3,
    )

    result = exploration.explore_signal(
        observed_cache=observed,
        episode_dim=episodes,
        contract_orientation_dim=orientations,
        config=config,
    )

    assert set(result["reaction_summary"]["candidate_label"]) == {"CASE_ONLY"}
    assert result["reaction_summary"]["game_cluster_ci_low"].isna().all()
    assert result["reaction_summary"]["loo_sign_stable"].isna().all()


def test_unknown_pre_trade_age_blocks_signal_candidate() -> None:
    observed, episodes, orientations = _frames()
    observed["signal_candidate_eligible"] = False
    observed["pre_trade_fresh_le_60s"] = pd.NA

    with pytest.raises(
        exploration.SignalExplorationError,
        match="no rows satisfy the frozen signal candidate analysis gate",
    ):
        exploration.explore_signal(
            observed_cache=observed,
            episode_dim=episodes,
            contract_orientation_dim=orientations,
            config=_config(),
        )


def test_optional_path_metrics_are_nullable_where_estimand_is_undefined() -> None:
    observed, episodes, orientations = _frames()
    observed.loc[observed.index[:3], "maximum_excursion"] = pd.NA
    observed.loc[observed.index[:8], "net_change_60s"] = pd.NA

    result = exploration.explore_signal(
        observed_cache=observed,
        episode_dim=episodes,
        contract_orientation_dim=orientations,
        config=_config(),
    )

    assert result["reaction_summary"]["signed_price_change_mean"].notna().all()


def test_non_focal_outcome_is_descriptive_only() -> None:
    observed, episodes, orientations = _frames()
    observed["signal_focal_team"] = "BBB"

    with pytest.raises(
        exploration.SignalExplorationError,
        match="signal_focal_team does not match",
    ):
        exploration.explore_signal(
            observed_cache=observed,
            episode_dim=episodes,
            contract_orientation_dim=orientations,
            config=_config(),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda observed, episodes, orientations: observed.assign(
                episode_id="orphan"
            ),
            "observed_cache references unknown episode_id",
        ),
        (
            lambda observed, episodes, orientations: pd.concat(
                [episodes, episodes.iloc[[0]]], ignore_index=True
            ),
            r"episode_dim game_id \+ episode_id must be unique",
        ),
        (
            lambda observed, episodes, orientations: observed.assign(
                delay_seconds=7
            ),
            "unregistered delay_seconds",
        ),
    ],
)
def test_referential_integrity_and_registration_fail_closed(
    mutation: object,
    match: str,
) -> None:
    observed, episodes, orientations = _frames()
    changed = mutation(observed, episodes, orientations)
    if match.startswith("episode_dim") or match.startswith(r"episode_dim"):
        episodes = changed
    else:
        observed = changed

    with pytest.raises(exploration.SignalExplorationError, match=match):
        exploration.explore_signal(
            observed_cache=observed,
            episode_dim=episodes,
            contract_orientation_dim=orientations,
            config=_config(),
        )


def test_extra_game_is_rejected_instead_of_truncated() -> None:
    observed, episodes, orientations = _frames(games=21)
    with pytest.raises(
        exploration.SignalExplorationError,
        match="exact frozen 20-game set",
    ):
        exploration.explore_signal(
            observed_cache=observed,
            episode_dim=episodes,
            contract_orientation_dim=orientations,
            config=_config(),
        )


def test_content_addressed_bundle_is_deterministic_and_detects_tampering(
    tmp_path: Path,
) -> None:
    observed, episodes, orientations = _frames()
    result = exploration.explore_signal(
        observed_cache=observed,
        episode_dim=episodes,
        contract_orientation_dim=orientations,
        config=_config(),
    )

    first = exploration.write_signal_exploration_bundle(
        result,
        output_root=tmp_path / "first",
        input_manifest_sha256="sha256:" + "a" * 64,
    )
    second = exploration.write_signal_exploration_bundle(
        result,
        output_root=tmp_path / "second",
        input_manifest_sha256="sha256:" + "a" * 64,
    )

    assert first.bundle_sha256 == second.bundle_sha256
    assert first.file_sha256 == second.file_sha256
    assert all(path.stem.removesuffix(".manifest") in digest for path, digest in [])
    exploration.verify_signal_exploration_bundle(first.manifest_path)
    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["runtime"]["network_used"] is False
    assert manifest["runtime"]["database_used"] is False
    assert (first.manifest_path.parent / "checksums.sha256").is_file()
    assert (first.manifest_path.parent / "X13_SIGNAL_REPORT.zh-CN.md").is_file()
    report_text = (
        first.manifest_path.parent / "X13_SIGNAL_REPORT.zh-CN.md"
    ).read_text()
    assert "逐场冻结因子" in report_text
    assert "ACTUAL_ONLY_NO_FORWARD_FILL" in report_text
    assert "真实成交观测" in report_text
    assert "缺失或无效单元" in report_text
    assert "fumble、turnover_on_downs" in report_text
    assert "不主张 OFI、depth" in report_text
    assert {record["logical_name"] for record in manifest["files"]} == {
        "report",
        "runtime_report",
        "coverage",
        "reaction_summary",
        "cross_venue_symmetry",
        "primary_contrasts",
        "path_comparisons",
        "state_moderators",
        "venue_comparisons",
        "actual_observations",
        "actual_observation_paths",
        "game_factor_table",
        "factor_coverage",
        "cross_game_factor_summary",
        "candidate_ranking",
        "per_game_market_coverage",
        "registered_hypothesis_results",
        "user_report_zh_cn",
    }
    for record in manifest["files"]:
        if record["logical_name"] != "user_report_zh_cn":
            assert record["sha256"].removeprefix("sha256:") in record["path"]

    summary_record = next(
        record
        for record in manifest["files"]
        if record["logical_name"] == "reaction_summary"
    )
    summary_path = first.manifest_path.parent / summary_record["path"]
    summary_path.write_bytes(summary_path.read_bytes() + b"tamper")
    with pytest.raises(
        exploration.SignalExplorationError,
        match="mismatch",
    ):
        exploration.verify_signal_exploration_bundle(first.manifest_path)


def test_runner_consumes_verified_joined_cache_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prediction_market.sports import nfl_x13_correlation_cache as cache

    observed, episodes, orientations = _frames()
    observed = observed.merge(
        episodes[["game_id", "episode_id", "episode_superclass"]],
        on=["game_id", "episode_id"],
        validate="many_to_one",
    )
    joined = observed.merge(
        orientations[
            [
                "contract_id",
                "outcome",
                "semantic_orientation_id",
                "orientation_audited",
            ]
        ],
        on=["contract_id", "outcome"],
        validate="many_to_one",
    ).rename(
        columns={
            "market_family": "family",
            "delay_seconds": "delay_scenario_seconds",
            "overshoot": "overshoot_candidate",
            "reversal": "reversal_candidate",
            "orientation_audited": "cross_venue_eligible",
        }
    )
    joined["validity_status"] = "OBSERVED"
    joined["order_ambiguous"] = False
    joined["contaminated"] = False
    joined["pre_trade_age_seconds"] = 1.0
    joined["pre_trade_age_status"] = (
        "RECONSTRUCTED_FROM_FROZEN_NORMALIZED_OBSERVATIONS"
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    observed_path = cache_root / "x13_reaction_observed_cache_v1.parquet"
    joined.to_parquet(observed_path, index=False)
    manifest_path = cache_root / "manifest.json"
    observed_sha = "sha256:" + hashlib.sha256(observed_path.read_bytes()).hexdigest()
    manifest = {
        "batch_id": (
            "sha256:3ddafb13fe58faa3caf01fa720a60f6c2efc4e87d9977b129b3c3824b43c1d85"
        ),
        "artifacts": {
            "observed_cache": {
                "relative_path": observed_path.name,
                "sha256": observed_sha,
                "byte_length": observed_path.stat().st_size,
            }
        }
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    monkeypatch.setattr(
        cache,
        "verify_x13_correlation_cache",
        lambda path: manifest if Path(path) == cache_root else {},
    )
    monkeypatch.setattr(
        exploration,
        "X13_GAME_IDS",
        tuple(sorted(joined["game_id"].unique())),
    )
    holdout_reference = {
        "status": "FROZEN",
        "selection_id": "sha256:" + "1" * 64,
        "selection_sha256": "sha256:" + "2" * 64,
        "byte_length": 100,
        "evidence_index_sha256": "sha256:" + "3" * 64,
        "selected_game_ids": [
            f"2025_{index + 21:02d}_CCC_DDD" for index in range(20)
        ],
    }
    monkeypatch.setattr(
        exploration,
        "_verify_historical_holdout_selection",
        lambda path: holdout_reference,
    )

    bundle = exploration.run_signal_exploration(
        cache_path=cache_root,
        holdout_selection_path=tmp_path / "holdout.json",
        output_root=tmp_path / "signal",
    )

    verified = exploration.verify_signal_exploration_bundle(
        bundle.manifest_path
    )
    assert verified["input_manifest_sha256"] == (
        "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    )
    assert verified["historical_holdout_selection"] == holdout_reference


@pytest.mark.parametrize("samples", [20, 9_999, 10_001, 10_000.0])
def test_bootstrap_sample_count_is_frozen(samples: object) -> None:
    with pytest.raises(
        exploration.SignalExplorationError,
        match="bootstrap_samples must equal 10000",
    ):
        exploration.SignalExplorationConfig(
            registered_delays_seconds=(0, 1, 2),
            registered_horizons_seconds=(5, 10, 30),
            bootstrap_samples=samples,
        )


def test_failed_second_bundle_publish_leaves_first_bundle_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed, episodes, orientations = _frames()
    result = exploration.explore_signal(
        observed_cache=observed,
        episode_dim=episodes,
        contract_orientation_dim=orientations,
        config=_config(),
    )
    root = tmp_path / "published"
    first = exploration.write_signal_exploration_bundle(
        result,
        output_root=root,
        input_manifest_sha256="sha256:" + "a" * 64,
    )
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    original = exploration._atomic_write
    calls = 0

    def fail_during_stage(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated stage failure")
        original(path, payload)

    monkeypatch.setattr(exploration, "_atomic_write", fail_during_stage)
    with pytest.raises(OSError, match="simulated stage failure"):
        exploration.write_signal_exploration_bundle(
            result,
            output_root=root,
            input_manifest_sha256="sha256:" + "b" * 64,
        )

    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
    exploration.verify_signal_exploration_bundle(first.manifest_path)


def test_binary_contribution_uses_contrast_influence_without_arm_cancellation() -> None:
    frame = pd.DataFrame(
        [
            {"game_id": "g1", "effect": 1.0, "arm": True},
            {"game_id": "g1", "effect": -1.0, "arm": False},
            {"game_id": "g2", "effect": 0.1, "arm": True},
            {"game_id": "g2", "effect": 0.0, "arm": False},
            {"game_id": "g3", "effect": 0.1, "arm": True},
            {"game_id": "g3", "effect": 0.0, "arm": False},
        ]
    )

    result = exploration._cluster_inference(
        frame,
        value="effect",
        arm="arm",
        config=_config(),
        seed_offset=99,
    )

    assert result["maximum_single_game_contribution"] > 0.4


def test_disappearing_bootstrap_arm_is_counted_invalid() -> None:
    frame = pd.DataFrame(
        [
            {"game_id": "g1", "effect": 1.0, "arm": True},
            {"game_id": "g2", "effect": 0.0, "arm": False},
        ]
    )

    result = exploration._cluster_inference(
        frame,
        value="effect",
        arm="arm",
        config=_config(),
        seed_offset=101,
    )

    assert result["bootstrap_samples_valid"] < 10_000
    assert result["logo_samples_valid"] < result["logo_samples_requested"]


def test_primary_support_gate_requires_each_arm() -> None:
    table = pd.DataFrame(
        [
            {
                "comparison_id": "score_minus_routine",
                "episode_count": 300,
                "game_count": 16,
                "positive_episode_count": 10,
                "negative_episode_count": 290,
                "positive_game_count": 5,
                "negative_game_count": 16,
                "point_estimate": 0.01,
                "ci_low": 0.006,
                "ci_high": 0.014,
                "raw_p_value": 0.001,
                "logo_sign_stability_rate": 1.0,
                "maximum_single_game_contribution": 0.1,
                "bootstrap_samples_requested": 10_000,
                "bootstrap_samples_valid": 10_000,
                "logo_samples_requested": 16,
                "logo_samples_valid": 16,
            }
        ]
    )

    labeled = exploration._label_frozen_table(
        table,
        support_kind="primary",
        config=_config(),
    )

    assert labeled.iloc[0]["candidate_label"] == "DESCRIPTIVE"


def test_atomic_write_fsyncs_file_and_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = exploration.os.fsync

    def record(descriptor: int) -> None:
        calls.append(descriptor)
        original(descriptor)

    monkeypatch.setattr(exploration.os, "fsync", record)
    exploration._atomic_write(tmp_path / "artifact.bin", b"verified")

    assert len(calls) == 2


def test_bounded_read_rejects_oversized_file_before_read(tmp_path: Path) -> None:
    path = tmp_path / "oversized.bin"
    path.write_bytes(b"12345")

    with pytest.raises(
        exploration.SignalExplorationError,
        match="byte length mismatch",
    ):
        exploration._read_bounded(path, max_length=4)


def test_frozen_factor_tables_are_per_game_and_actual_observation_only() -> None:
    observed, episodes, orientations = _frames()
    seed = observed[
        observed["delay_seconds"].eq(0)
        & observed["horizon_seconds"].eq(10)
    ].copy()
    expanded: list[pd.DataFrame] = []
    for horizon in (1, 2, 5, 10, 30, 60):
        frame = seed.copy()
        frame["horizon_seconds"] = horizon
        frame["signed_price_change"] = (
            frame["signed_price_change"] * (horizon / 10)
        )
        frame["pre_event_actual_trade"] = "0.5000"
        expanded.append(frame)
    observed = pd.concat(expanded, ignore_index=True)
    missing = (
        observed["game_id"].eq("2025_01_AAA_BBB")
        & observed["episode_id"].eq("2025_01_AAA_BBB:episode:0")
        & observed["contract_id"].eq("kalshi:AAA")
        & observed["horizon_seconds"].eq(2)
    )
    observed = observed[~missing].reset_index(drop=True)
    event_factors = episodes[["game_id", "episode_id"]].copy()
    event_factors["interception"] = event_factors["episode_id"].str.endswith(":1")
    event_factors["blocked_kick"] = False
    event_factors["penalty"] = False
    event_factors["review"] = False
    event_factors["timeout"] = False
    event_factors["fumble"] = pd.NA
    event_factors["turnover_on_downs"] = pd.NA
    event_factors["lineage_build_id"] = "sha256:" + "b" * 64

    result = exploration.explore_signal(
        observed_cache=observed,
        episode_dim=episodes,
        contract_orientation_dim=orientations,
        config=exploration.SignalExplorationConfig(
            registered_delays_seconds=(0,),
            registered_horizons_seconds=(1, 2, 5, 10, 30, 60),
            bootstrap_samples=10_000,
            minimum_episode_count=4,
            minimum_game_count=3,
        ),
        event_factor_dim=event_factors,
    )

    actual = result["actual_observations"]
    assert set(actual["horizon_seconds"]) == {1, 2, 5, 10, 30, 60}
    assert actual["actual_observation"].eq(True).all()
    assert actual["forward_filled"].eq(False).all()
    assert not (
        actual["game_id"].eq("2025_01_AAA_BBB")
        & actual["episode_id"].eq("2025_01_AAA_BBB:episode:0")
        & actual["contract_id"].eq("kalshi:AAA")
        & actual["horizon_seconds"].eq(2)
    ).any()

    per_game = result["game_factor_table"]
    assert set(per_game["game_id"]) == set(exploration.X13_GAME_IDS)
    assert set(per_game["factor_domain"]) == {
        "EVENT",
        "STATE",
        "COMBINATION",
        "MARKET_REGIME",
        "PATH",
    }
    assert per_game["source_delay_seconds"].eq(0).all()
    assert per_game["observation_policy"].eq(
        "ACTUAL_ONLY_NO_FORWARD_FILL"
    ).all()

    paths = result["actual_observation_paths"]
    incomplete = paths[
        paths["game_id"].eq("2025_01_AAA_BBB")
        & paths["episode_id"].eq("2025_01_AAA_BBB:episode:0")
        & paths["contract_id"].eq("kalshi:AAA")
    ]
    assert not incomplete.empty
    assert incomplete["change_2s"].isna().all()
    assert not incomplete["path_complete"].any()
    assert incomplete["path_factor_value"].eq("INCOMPLETE_ACTUAL_PATH").all()

    coverage = result["factor_coverage"]
    interception = coverage[
        coverage["factor_domain"].eq("EVENT")
        & coverage["factor_key"].eq("interception")
    ].iloc[0]
    assert interception["availability_status"] == (
        "AVAILABLE_BOUND_STRUCTURED_FIELD"
    )
    for unavailable_key in ("fumble", "turnover_on_downs"):
        unavailable = coverage[
            coverage["factor_domain"].eq("EVENT")
            & coverage["factor_key"].eq(unavailable_key)
        ].iloc[0]
        assert unavailable["availability_status"] == "UNAVAILABLE_SOURCE_FIELD"
        assert unavailable["observation_count"] == 0

    summary = result["cross_game_factor_summary"]
    ranking = result["candidate_ranking"]
    assert {"game_count", "cross_game_mean", "candidate_label"} <= set(summary)
    assert {"candidate_rank", "frozen_factor_id", "candidate_label"} <= set(
        ranking
    )
    assert {
        "point_estimate",
        "ci_low",
        "ci_high",
        "logo_sign_stability_rate",
        "maximum_single_game_contribution",
        "raw_p_value",
        "bh_q_value",
        "kalshi_effect",
        "polymarket_effect",
        "cross_venue_sign_consistent",
        "support_gate_pass",
    } <= set(ranking)
    assert set(ranking["candidate_label"]) == {"DISCOVERY_ONLY"}
    assert "SIGNAL_CANDIDATE" not in set(ranking["candidate_label"])
    assert "ABSENT" not in set(ranking["factor_value"])
    assert "INCOMPLETE_ACTUAL_PATH" not in set(ranking["factor_value"])
    assert not (
        ranking["factor_domain"].eq("MARKET_REGIME")
        & ranking["factor_key"].isin(["venue", "family"])
    ).any()


def test_publication_retains_invalid_cells_but_factor_analysis_is_strict() -> None:
    observed, episodes, orientations = _frames()
    invalid_index = observed[
        observed["delay_seconds"].eq(0)
        & observed["horizon_seconds"].eq(10)
    ].index[0]
    observed.loc[invalid_index, "validity_status"] = "NO_POST_TRADE"
    observed.loc[invalid_index, "order_ambiguous"] = True
    observed.loc[invalid_index, "signal_candidate_eligible"] = False
    observed.loc[invalid_index, "trade_count"] = 0
    observed.loc[invalid_index, "volume"] = 0
    observed.loc[
        invalid_index,
        [
            "signed_price_change",
            "first_post_event_trade",
            "vwap",
            "maximum_excursion",
            "net_change_60s",
        ],
    ] = pd.NA

    result = exploration.explore_signal(
        observed_cache=observed,
        episode_dim=episodes,
        contract_orientation_dim=orientations,
        config=_config(),
    )

    actual = result["actual_observations"]
    invalid = actual[
        actual["game_id"].eq(observed.loc[invalid_index, "game_id"])
        & actual["episode_id"].eq(observed.loc[invalid_index, "episode_id"])
        & actual["contract_id"].eq(observed.loc[invalid_index, "contract_id"])
        & actual["horizon_seconds"].eq(10)
    ]
    assert len(invalid) == 1
    assert invalid.iloc[0]["validity_status"] == "NO_POST_TRADE"
    assert bool(invalid.iloc[0]["order_ambiguous"]) is True
    assert bool(invalid.iloc[0]["actual_observation"]) is False
    assert bool(invalid.iloc[0]["analysis_eligible"]) is False
    assert pd.isna(invalid.iloc[0]["last_actual_trade"])
    assert {
        "source_time_start_utc",
        "source_time_end_utc",
        "pre_event_actual_trade",
        "first_post_event_trade",
        "vwap",
        "last_actual_trade",
        "maximum_excursion",
        "staleness_seconds",
        "contaminated",
        "order_ambiguous",
        "validity_status",
        "analysis_eligible",
        "analysis_exclusion_reason",
    } <= set(actual)
    assert result["candidate_ranking"]["candidate_label"].eq(
        "DISCOVERY_ONLY"
    ).all()


def test_ineligible_event_may_have_no_focal_team_without_entering_analysis() -> None:
    observed, episodes, orientations = _frames()
    ineligible = observed.index[0]
    observed.loc[ineligible, "signal_event_eligible"] = False
    observed.loc[ineligible, "signal_candidate_eligible"] = False
    observed.loc[ineligible, "signal_focal_team"] = None

    result = exploration.explore_signal(
        observed_cache=observed,
        episode_dim=episodes,
        contract_orientation_dim=orientations,
        config=_config(),
    )

    actual = result["actual_observations"]
    row = actual[
        actual["game_id"].eq(observed.loc[ineligible, "game_id"])
        & actual["episode_id"].eq(observed.loc[ineligible, "episode_id"])
        & actual["contract_id"].eq(observed.loc[ineligible, "contract_id"])
        & actual["horizon_seconds"].eq(
            observed.loc[ineligible, "horizon_seconds"]
        )
    ]
    assert len(row) == 1
    assert bool(row.iloc[0]["analysis_eligible"]) is False
    assert row.iloc[0]["analysis_exclusion_reason"] == (
        "SIGNAL_CANDIDATE_INELIGIBLE"
    )


def test_frozen_factor_catalog_is_complete_and_fail_closed() -> None:
    observed, episodes, orientations = _frames()
    observed["episode_type"] = observed["episode_id"].map(
        lambda value: (
            "touchdown"
            if value.endswith(":0")
            else "pass"
            if value.endswith(":1")
            else "punt"
        )
    )
    observed["source_time_start_utc"] = observed["episode_id"].map(
        lambda value: (
            "2025-09-01T00:00:01Z"
            if value.endswith(":0")
            else "2025-09-01T00:00:02Z"
            if value.endswith(":1")
            else "2025-09-01T00:00:03Z"
        )
    )
    observed["pre_possession_team"] = observed["pre_offense_team"]
    observed["pre_distance"] = 7
    observed["pre_goal_to_go"] = False
    observed["pre_event_actual_trade"] = "0.5000"

    result = exploration.explore_signal(
        observed_cache=observed,
        episode_dim=episodes,
        contract_orientation_dim=orientations,
        config=_config(),
    )

    coverage = result["factor_coverage"].set_index(
        ["factor_domain", "factor_key"]
    )
    required = {
        ("EVENT", "touchdown"),
        ("EVENT", "field_goal"),
        ("EVENT", "punt"),
        ("EVENT", "safety"),
        ("EVENT", "score_turnover"),
        ("EVENT", "pat"),
        ("EVENT", "two_point_conversion"),
        ("EVENT", "scoring_reversal"),
        ("EVENT", "fumble"),
        ("EVENT", "turnover_on_downs"),
        ("STATE", "time_remaining_band"),
        ("STATE", "possession_side"),
        ("STATE", "offense_side"),
        ("STATE", "distance_band"),
        ("STATE", "goal_to_go"),
        ("STATE", "red_zone"),
        ("STATE", "staleness_band"),
        ("STATE", "recent_activity_band"),
        ("STATE", "timeouts_remaining"),
        ("STATE", "drive_progression"),
        ("STATE", "win_probability_band"),
        ("COMBINATION", "late_close_score"),
        ("COMBINATION", "red_zone_turnover"),
        ("COMBINATION", "third_fourth_down_failure"),
        ("COMBINATION", "answer_score"),
        ("COMBINATION", "back_to_back_turnover"),
        ("COMBINATION", "two_minute_drive"),
        ("COMBINATION", "overtime"),
        ("MARKET_REGIME", "pre_event_probability_band"),
        ("MARKET_REGIME", "recent_activity_band"),
        ("MARKET_REGIME", "staleness_band"),
        ("MARKET_REGIME", "venue"),
        ("MARKET_REGIME", "family"),
    }
    assert required <= set(coverage.index)

    for identity in {
        ("EVENT", "pat"),
        ("EVENT", "two_point_conversion"),
        ("EVENT", "scoring_reversal"),
        ("EVENT", "fumble"),
        ("EVENT", "turnover_on_downs"),
        ("STATE", "recent_activity_band"),
        ("STATE", "timeouts_remaining"),
        ("STATE", "drive_progression"),
        ("STATE", "win_probability_band"),
        ("COMBINATION", "third_fourth_down_failure"),
        ("COMBINATION", "two_minute_drive"),
        ("MARKET_REGIME", "recent_activity_band"),
    }:
        assert coverage.loc[identity, "availability_status"] == (
            "UNAVAILABLE_SOURCE_FIELD"
        )
        assert coverage.loc[identity, "observation_count"] == 0

    hypotheses = result["registered_hypothesis_results"].set_index(
        ["factor_domain", "factor_key"]
    )
    assert required <= set(hypotheses.index)
    assert hypotheses.loc[
        ("COMBINATION", "two_minute_drive"), "run_status"
    ] == "NOT_RUN"
    assert hypotheses.loc[
        ("COMBINATION", "two_minute_drive"), "status_reason"
    ] == "UNAVAILABLE_SOURCE_FIELD"
    diagnostic_fair = hypotheses[
        hypotheses["hypothesis_id"].str.startswith("fair_delta_")
        | hypotheses["hypothesis_id"].str.startswith(
            "signed_diagnostic_fair_delta_"
        )
    ]
    assert diagnostic_fair["run_status"].eq("NOT_RUN").all()
    assert diagnostic_fair["status_reason"].eq(
        "NO_PIT_PROVEN_DIAGNOSTIC_FAIR_DELTA"
    ).all()


def test_bound_game_state_lineage_uses_only_structured_event_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game_id = "2025_01_AAA_BBB"
    monkeypatch.setattr(exploration, "X13_GAME_IDS", (game_id,))
    batch_id = "sha256:" + "b" * 64
    build_id = "sha256:" + "c" * 64
    x13_root = tmp_path / "x13"
    cache_root = x13_root / "correlation-cache" / "sha256-cache"
    cache_root.mkdir(parents=True)
    batch_root = x13_root / "batches" / batch_id.replace(":", "-")
    (batch_root / "reports").mkdir(parents=True)
    lineage_path = batch_root / "reports" / "lineage-report-v1.json"
    lineage_payload = json.dumps(
        {"game_state_build_id": build_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    lineage_path.write_bytes(lineage_payload)
    batch_manifest = {
        "batch_id": batch_id,
        "auxiliary_artifacts": [
            {
                "relative_path": "reports/lineage-report-v1.json",
                "sha256": "sha256:"
                + hashlib.sha256(lineage_payload).hexdigest(),
                "byte_length": len(lineage_payload),
            }
        ],
    }
    (batch_root / "batch_manifest.json").write_text(
        json.dumps(batch_manifest, sort_keys=True, separators=(",", ":"))
    )

    state_root = x13_root / "game-state" / build_id.replace(":", "-")
    game_root = state_root / "games" / game_id
    game_root.mkdir(parents=True)
    build_payload = json.dumps(
        {"build_id": build_id, "game_ids": [game_id]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    episodes_payload = json.dumps(
        [
            {
                "game_id": game_id,
                "episode_id": "episode-1",
                "play_ids": ["play-1"],
            }
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    events_payload = json.dumps(
        [
            {
                "game_id": game_id,
                "play_id": "play-1",
                "player_roles": {"interceptor": "player"},
                "blocked_kick": False,
                "penalty": True,
                "review": False,
                "timeout": True,
                "description": "FUMBLES and recovered",
            }
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    files = {
        "build_manifest.json": build_payload,
        f"games/{game_id}/finalized_episodes.json": episodes_payload,
        f"games/{game_id}/canonical_events.json": events_payload,
    }
    for relative, payload in files.items():
        target = state_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (state_root / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(payload).hexdigest()}  {relative}\n"
            for relative, payload in sorted(files.items())
        )
    )

    factors = exploration._load_bound_event_factor_dim(
        cache_root=cache_root,
        batch_id=batch_id,
    )

    assert factors.to_dict("records")[0]["interception"] is True
    assert factors.to_dict("records")[0]["penalty"] is True
    assert factors.to_dict("records")[0]["timeout"] is True
    assert pd.isna(factors.iloc[0]["fumble"])
    assert pd.isna(factors.iloc[0]["turnover_on_downs"])
