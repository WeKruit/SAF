from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest

from prediction_market.contracts import EventEnvelopeV0
from prediction_market.models import nfl
from prediction_market.sports.event_envelopes import (
    StaticSportObservationEnvelopeBundle,
    build_static_sport_observation_bundle,
)
from prediction_market.sports.nfl_game_state import NFLGameState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GAME_ID = "game_nflverse_2025_01_AWY_HME"
_STATE_SOURCE_AT = "2025-09-07T19:30:00Z"
_RECEIVER_OBSERVED_AT = datetime(2025, 9, 7, 17, 55, tzinfo=timezone.utc)
_SPREAD_OBSERVED_AT = datetime(2025, 9, 7, 19, 0, tzinfo=timezone.utc)
_MANIFEST_SHA256 = "sha256:" + "b" * 64
_OBJECT_SHA256 = "sha256:" + "7" * 64


def _state(**changes: object) -> NFLGameState:
    values: dict[str, object] = {
        "sport": "nfl",
        "game_id": _GAME_ID,
        "sequence": 12,
        "terminal": False,
        "home_team": "HME",
        "away_team": "AWY",
        "period": 2,
        "period_seconds_remaining": 600,
        "game_seconds_remaining": 2400,
        "source_play_id": "101",
        "drive_id": "3",
        "play_clock_seconds": 12,
        "possession_team": "HME",
        "down": 2,
        "distance": 4,
        "yardline_100": 45,
        "goal_to_go": False,
        "home_score": 14,
        "away_score": 7,
        "home_timeouts_remaining": 2,
        "away_timeouts_remaining": 1,
        "last_event_id": None,
    }
    values.update(changes)
    return NFLGameState(**values)


def _event_payload(
    state: NFLGameState,
    *,
    game_id: str,
    changes: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "sport": "nfl",
        "game_id": game_id,
        "sequence": state.sequence,
        "source_play_id": "100",
        "observation_mode": "offline",
        "play_type": "run",
        "description": "fixture play",
        "period": state.period,
        "period_seconds_remaining": state.period_seconds_remaining,
        "game_seconds_remaining": state.game_seconds_remaining,
        "next_source_play_id": state.source_play_id,
        "next_drive_id": state.drive_id,
        "next_play_clock_seconds": state.play_clock_seconds,
        "possession_team": state.possession_team,
        "down": state.down,
        "distance": state.distance,
        "yardline_100": state.yardline_100,
        "goal_to_go": state.goal_to_go,
        "home_score": state.home_score,
        "away_score": state.away_score,
        "home_timeouts_remaining": state.home_timeouts_remaining,
        "away_timeouts_remaining": state.away_timeouts_remaining,
        "first_down": False,
        "turnover": False,
        "possession_changed": False,
        "score": False,
        "timeout": False,
        "timeout_team": None,
        "carry_forward_context": False,
        "period_changed": False,
        "terminal": state.terminal,
        "quality_flags": [],
    }
    if changes is not None:
        payload.update(changes)
    return payload


def _state_envelope_bundle(
    state: NFLGameState,
    *,
    game_id: str | None = None,
    source_at: str | None = _STATE_SOURCE_AT,
    raw_object_hash: str = "sha256:" + "7" * 64,
    payload_changes: dict[str, object] | None = None,
) -> StaticSportObservationEnvelopeBundle:
    canonical_game_id = state.game_id if game_id is None else game_id
    native_game_id = canonical_game_id.removeprefix("game_nflverse_")
    return build_static_sport_observation_bundle(
        program_root=PROJECT_ROOT,
        experiment_id="X-11",
        dataset_id="DS-NFLVERSE",
        source_system="nflverse",
        source_stream="play_by_play",
        raw_object_hash=raw_object_hash,
        raw_record_ordinals=(10, 11),
        partition="season-2025",
        fetched_at="2026-07-22T12:00:00Z",
        source_at=source_at,
        competition_id="cmp_nfl",
        game_id=canonical_game_id,
        participant_ids=(
            "participant_nflverse_AWY",
            "participant_nflverse_HME",
        ),
        native_namespace="nflverse.play",
        native_ids=(
            f"{native_game_id}:100",
            f"{native_game_id}:{state.source_play_id}",
        ),
        normalized_source_sequence=state.sequence,
        normalized_payload=_event_payload(
            state,
            game_id=canonical_game_id,
            changes=payload_changes,
        ),
    )


def _recreate_envelope(
    envelope: EventEnvelopeV0,
    **changes: object,
) -> EventEnvelopeV0:
    material = envelope.model_dump(mode="python", round_trip=True)
    material.pop("event_id")
    material.pop("payload_sha256")
    material.update(changes)
    return EventEnvelopeV0.create(**material)


def _relink_normalized_parent_ids(
    envelope: EventEnvelopeV0,
    parents: tuple[EventEnvelopeV0, ...],
) -> EventEnvelopeV0:
    lineage = envelope.lineage.model_dump(mode="python")
    lineage["parent_event_ids"] = tuple(
        sorted(parent.event_id for parent in parents)
    )
    return _recreate_envelope(envelope, lineage=lineage)


def _context(
    *,
    state_event_envelope: object,
    **changes: object,
) -> nfl.NFLWinProbabilityContext:
    values: dict[str, object] = {
        "state_event_envelope": state_event_envelope,
        "second_half_receiver": "HME",
        "second_half_receiver_source_ref": "nflverse_pbp:receive_2h_ko",
        "second_half_receiver_observed_at": _RECEIVER_OBSERVED_AT,
        "home_spread_line": -3.5,
        "spread_observed_at": _SPREAD_OBSERVED_AT,
        "declared_source_manifest_sha256": _MANIFEST_SHA256,
        "source_object_sha256": _OBJECT_SHA256,
        "pit_status": "offline_reconstruction_not_live_PIT",
    }
    values.update(changes)
    return nfl.NFLWinProbabilityContext(**values)


def _fixture(
    *,
    state_changes: dict[str, object] | None = None,
    context_changes: dict[str, object] | None = None,
    bundle_game_id: str | None = None,
    source_at: str | None = _STATE_SOURCE_AT,
    raw_object_hash: str = "sha256:" + "7" * 64,
) -> tuple[
    NFLGameState,
    nfl.NFLWinProbabilityContext,
    StaticSportObservationEnvelopeBundle,
]:
    provisional = _state(**(state_changes or {}))
    bundle = _state_envelope_bundle(
        provisional,
        game_id=bundle_game_id,
        source_at=source_at,
        raw_object_hash=raw_object_hash,
    )
    state = replace(
        provisional,
        last_event_id=bundle.normalized.event_id,
    )
    resolved_context_changes = {
        "source_object_sha256": raw_object_hash,
        **(context_changes or {}),
    }
    context = _context(
        state_event_envelope=bundle.normalized,
        **resolved_context_changes,
    )
    return state, context, bundle


def _feature_vector(
    state: NFLGameState,
    context: nfl.NFLWinProbabilityContext,
    bundle: StaticSportObservationEnvelopeBundle,
    *,
    variant: Literal["no_spread", "spread"],
) -> nfl.FastrmodelsFeatureVector:
    return nfl.fastrmodels_feature_vector(
        state,
        context=context,
        variant=variant,
        program_root=PROJECT_ROOT,
        state_event_raw_parents=bundle.raw,
    )


def test_official_feature_order_and_transform_are_frozen() -> None:
    state, context, bundle = _fixture()

    no_spread = _feature_vector(
        state,
        context,
        bundle,
        variant="no_spread",
    )
    spread = _feature_vector(
        state,
        context,
        bundle,
        variant="spread",
    )

    elapsed_share = (3600 - 2400) / 3600
    decay = math.exp(-4 * elapsed_share)
    assert no_spread.names == nfl.FASTRMODELS_NO_SPREAD_FEATURES
    assert no_spread.values == pytest.approx(
        (
            1.0,
            1.0,
            600.0,
            2400.0,
            7.0 / decay,
            7.0,
            2.0,
            4.0,
            45.0,
            2.0,
            1.0,
        )
    )
    assert spread.names == nfl.FASTRMODELS_SPREAD_FEATURES
    assert spread.values == pytest.approx(
        (
            1.0,
            -3.5 * decay,
            1.0,
            600.0,
            2400.0,
            7.0 / decay,
            7.0,
            2.0,
            4.0,
            45.0,
            2.0,
            1.0,
        )
    )
    assert no_spread.source_state_sha256 == (
        "sha256:dd2146068ba3d3fa3ff22c6dd25d05a6c82acd85017b8c4f5c6afbbdf587c845"
    )
    assert no_spread.feature_sha256 == (
        "sha256:a3907f042f2b7f7b32e77e59a54bb8671d73d9bdd93eba7c18f93a380b8bc014"
    )
    assert spread.feature_sha256 == (
        "sha256:0e18cf6a4bd0d1875bdfa4fa503fc345256410c86fa4dfa2316321afce7c4e36"
    )
    assert (
        no_spread.model_artifact_sha256
        == nfl.FASTRMODELS_NO_SPREAD_MODEL_SHA256
    )
    assert (
        spread.model_artifact_sha256
        == nfl.FASTRMODELS_SPREAD_MODEL_SHA256
    )
    assert spread.feature_sha256 != no_spread.feature_sha256


def test_away_possession_orients_score_spread_and_timeouts() -> None:
    state, context, bundle = _fixture(
        state_changes={"possession_team": "AWY"},
        context_changes={"second_half_receiver": "AWY"},
    )

    features = _feature_vector(
        state,
        context,
        bundle,
        variant="spread",
    ).as_dict()

    assert features["receive_2h_ko"] == 1.0
    assert features["home"] == 0.0
    assert features["score_differential"] == -7.0
    assert features["posteam_timeouts_remaining"] == 1.0
    assert features["defteam_timeouts_remaining"] == 2.0
    assert features["spread_time"] > 0.0


@pytest.mark.parametrize(
    ("state_changes", "context_changes", "variant", "message"),
    [
        (
            {
                "period": 5,
                "period_seconds_remaining": 600,
                "game_seconds_remaining": 600,
            },
            {},
            "spread",
            "regulation",
        ),
        (
            {"possession_team": None, "down": None, "distance": None},
            {},
            "spread",
            "possession",
        ),
        (
            {},
            {"home_spread_line": None, "spread_observed_at": None},
            "spread",
            "spread",
        ),
        (
            {},
            {"second_half_receiver": "XXX"},
            "spread",
            "receiver",
        ),
        (
            {
                "period": 4,
                "period_seconds_remaining": 0,
                "game_seconds_remaining": 0,
                "terminal": True,
            },
            {},
            "spread",
            "terminal",
        ),
    ],
)
def test_official_features_fail_closed_on_ineligible_state(
    state_changes: dict[str, object],
    context_changes: dict[str, object],
    variant: Literal["no_spread", "spread"],
    message: str,
) -> None:
    state, context, bundle = _fixture(
        state_changes=state_changes,
        context_changes=context_changes,
    )
    with pytest.raises(nfl.NFLModelInputError, match=message):
        _feature_vector(
            state,
            context,
            bundle,
            variant=variant,
        )


def test_second_half_receiver_flag_is_zero_after_halftime() -> None:
    state, context, bundle = _fixture(
        state_changes={
            "period": 3,
            "period_seconds_remaining": 600,
            "game_seconds_remaining": 1500,
        },
        context_changes={
            "home_spread_line": None,
            "spread_observed_at": None,
        },
    )

    features = _feature_vector(
        state,
        context,
        bundle,
        variant="no_spread",
    ).as_dict()

    assert features["receive_2h_ko"] == 0.0
    assert features["half_seconds_remaining"] == 1500.0


def test_feature_digest_binds_context_envelope_raw_lineage_and_variant() -> None:
    state, context, bundle = _fixture()
    baseline = _feature_vector(
        state,
        context,
        bundle,
        variant="spread",
    )

    contexts = (
        replace(
            context,
            second_half_receiver_source_ref="nflverse_pbp:coin_toss",
        ),
        replace(
            context,
            second_half_receiver_observed_at=(
                context.second_half_receiver_observed_at - timedelta(seconds=1)
            ),
        ),
        replace(
            context,
            spread_observed_at=context.spread_observed_at - timedelta(seconds=1),
        ),
        replace(
            context,
            declared_source_manifest_sha256="sha256:" + "d" * 64,
        ),
        replace(context, pit_status="PIT_UNPROVEN"),
    )
    digests = {
        _feature_vector(
            state,
            candidate,
            bundle,
            variant="spread",
        ).feature_sha256
        for candidate in contexts
    }
    moved_game_id = "game_nflverse_2025_02_AWY_HME"
    moved_game, moved_game_context, moved_game_bundle = _fixture(
        state_changes={"game_id": moved_game_id},
        bundle_game_id=moved_game_id,
    )
    moved_source, moved_source_context, moved_source_bundle = _fixture(
        source_at="2025-09-07T19:30:00.000001Z",
    )
    moved_raw, moved_raw_context, moved_raw_bundle = _fixture(
        raw_object_hash="sha256:" + "9" * 64,
    )
    no_spread = _feature_vector(
        state,
        context,
        bundle,
        variant="no_spread",
    )

    assert baseline.feature_sha256 not in digests
    assert len(digests) == len(contexts)
    assert _feature_vector(
        moved_game,
        moved_game_context,
        moved_game_bundle,
        variant="spread",
    ).feature_sha256 != baseline.feature_sha256
    assert _feature_vector(
        moved_source,
        moved_source_context,
        moved_source_bundle,
        variant="spread",
    ).feature_sha256 != baseline.feature_sha256
    assert _feature_vector(
        moved_raw,
        moved_raw_context,
        moved_raw_bundle,
        variant="spread",
    ).feature_sha256 != baseline.feature_sha256
    assert no_spread.feature_sha256 != baseline.feature_sha256


def test_feature_digest_binds_selected_official_model_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, context, bundle = _fixture()
    baseline = _feature_vector(
        state,
        context,
        bundle,
        variant="spread",
    )
    replacement_sha256 = "sha256:" + "f" * 64

    monkeypatch.setattr(
        nfl,
        "FASTRMODELS_SPREAD_MODEL_SHA256",
        replacement_sha256,
    )
    changed = _feature_vector(
        state,
        context,
        bundle,
        variant="spread",
    )

    assert changed.model_artifact_sha256 == replacement_sha256
    assert changed.names == baseline.names
    assert changed.values == baseline.values
    assert changed.feature_sha256 != baseline.feature_sha256


def test_post_cutoff_external_observation_fails_closed() -> None:
    state, context, bundle = _fixture()
    post_cutoff = datetime(
        2025,
        9,
        7,
        19,
        30,
        0,
        1,
        tzinfo=timezone.utc,
    )

    with pytest.raises(nfl.NFLModelInputError, match="spread_observed_at"):
        _feature_vector(
            state,
            replace(context, spread_observed_at=post_cutoff),
            bundle,
            variant="spread",
        )

    with pytest.raises(
        nfl.NFLModelInputError,
        match="second_half_receiver_observed_at",
    ):
        _feature_vector(
            state,
            replace(
                context,
                second_half_receiver_observed_at=post_cutoff,
            ),
            bundle,
            variant="no_spread",
        )


def test_envelope_game_must_match_reducer_state() -> None:
    state, context, bundle = _fixture(
        bundle_game_id="game_nflverse_2025_02_AWY_HME",
    )
    with pytest.raises(nfl.NFLModelInputError, match="game_id"):
        _feature_vector(
            state,
            context,
            bundle,
            variant="spread",
        )


def test_envelope_event_id_must_match_reducer_state() -> None:
    state, context, bundle = _fixture()
    state = replace(state, last_event_id="evt_" + "d" * 64)

    with pytest.raises(nfl.NFLModelInputError, match="event_id"):
        _feature_vector(
            state,
            context,
            bundle,
            variant="spread",
        )


def test_envelope_source_play_must_match_reducer_state() -> None:
    state, context, bundle = _fixture()
    state = replace(state, source_play_id="999")

    with pytest.raises(nfl.NFLModelInputError, match="source_play_id"):
        _feature_vector(
            state,
            context,
            bundle,
            variant="spread",
        )


def test_envelope_sequence_must_match_reducer_state() -> None:
    state, context, bundle = _fixture()
    state = replace(state, sequence=13)

    with pytest.raises(nfl.NFLModelInputError, match="sequence"):
        _feature_vector(
            state,
            context,
            bundle,
            variant="spread",
        )


@pytest.mark.parametrize(
    ("state_changes", "field"),
    [
        (
            {
                "terminal": True,
                "period": 4,
                "period_seconds_remaining": 0,
                "game_seconds_remaining": 0,
            },
            "terminal",
        ),
        (
            {
                "period": 4,
                "period_seconds_remaining": 600,
                "game_seconds_remaining": 600,
            },
            "period",
        ),
        (
            {
                "period_seconds_remaining": 599,
                "game_seconds_remaining": 2399,
            },
            "period_seconds_remaining",
        ),
        ({"drive_id": "4"}, "drive_id"),
        ({"play_clock_seconds": 13}, "play_clock_seconds"),
        ({"possession_team": "AWY"}, "possession_team"),
        ({"down": 3}, "down"),
        ({"distance": 5}, "distance"),
        ({"yardline_100": 44}, "yardline_100"),
        ({"goal_to_go": True}, "goal_to_go"),
        ({"home_score": 15}, "home_score"),
        ({"away_score": 8}, "away_score"),
        ({"home_timeouts_remaining": 1}, "home_timeouts_remaining"),
        ({"away_timeouts_remaining": 2}, "away_timeouts_remaining"),
    ],
)
def test_envelope_rejects_mutated_post_state_projection(
    state_changes: dict[str, object],
    field: str,
) -> None:
    state, context, bundle = _fixture()
    state = replace(state, **state_changes)

    with pytest.raises(nfl.NFLModelInputError, match=field):
        _feature_vector(
            state,
            context,
            bundle,
            variant="spread",
        )


def test_envelope_participants_must_match_state_teams() -> None:
    state, context, bundle = _fixture()
    changed_refs = bundle.normalized.canonical_refs.model_dump(mode="python")
    changed_refs["participant_ids"] = (
        "participant_nflverse_ALT",
        "participant_nflverse_HME",
    )
    parents = tuple(
        _recreate_envelope(parent, canonical_refs=changed_refs)
        for parent in bundle.raw
    )
    lineage = bundle.normalized.lineage.model_dump(mode="python")
    lineage["parent_event_ids"] = tuple(
        parent.event_id for parent in parents
    )
    normalized = _recreate_envelope(
        bundle.normalized,
        canonical_refs=changed_refs,
        lineage=lineage,
    )
    state = replace(state, last_event_id=normalized.event_id)
    context = replace(context, state_event_envelope=normalized)

    with pytest.raises(nfl.NFLModelInputError, match="participant_ids"):
        nfl.fastrmodels_feature_vector(
            state,
            context=context,
            variant="spread",
            program_root=PROJECT_ROOT,
            state_event_raw_parents=parents,
        )


def test_envelope_rejects_home_away_role_swap_with_same_participant_set() -> None:
    state, context, bundle = _fixture()
    state = replace(
        state,
        home_team="AWY",
        away_team="HME",
    )

    with pytest.raises(
        nfl.NFLModelInputError,
        match="home_team|away_team",
    ):
        _feature_vector(
            state,
            context,
            bundle,
            variant="spread",
        )


def test_coherently_forged_2020_envelope_cannot_describe_2025_game() -> None:
    old_receiver_at = datetime(2020, 9, 7, 17, 55, tzinfo=timezone.utc)
    old_spread_at = datetime(2020, 9, 7, 19, 0, tzinfo=timezone.utc)
    state, context, bundle = _fixture(
        source_at="2020-09-07T19:30:00Z",
        context_changes={
            "second_half_receiver_observed_at": old_receiver_at,
            "spread_observed_at": old_spread_at,
        },
    )

    with pytest.raises(nfl.NFLModelInputError, match="season"):
        _feature_vector(
            state,
            context,
            bundle,
            variant="spread",
        )


def test_source_at_is_required_as_the_only_offline_state_cutoff() -> None:
    state, context, bundle = _fixture(source_at=None)

    with pytest.raises(nfl.NFLModelInputError, match="source_at"):
        _feature_vector(
            state,
            context,
            bundle,
            variant="spread",
        )


def test_offline_state_envelope_cannot_claim_live_pit_status() -> None:
    state, context, bundle = _fixture(
        context_changes={"pit_status": "live_pit"},
    )

    with pytest.raises(nfl.NFLModelInputError, match="pit_status"):
        _feature_vector(
            state,
            context,
            bundle,
            variant="spread",
        )


def test_tampered_state_envelope_is_revalidated_at_feature_seam() -> None:
    state, context, bundle = _fixture()
    tampered_time = context.state_event_envelope.time.model_copy(
        update={"source_at": "2025-09-07T19:30:00.000001Z"}
    )
    tampered = context.state_event_envelope.model_copy(
        update={"time": tampered_time}
    )

    with pytest.raises(
        nfl.NFLModelInputError,
        match="state event envelope",
    ):
        _feature_vector(
            state,
            replace(context, state_event_envelope=tampered),
            bundle,
            variant="spread",
        )


def test_tampered_raw_parent_is_revalidated_at_feature_seam() -> None:
    state, context, bundle = _fixture()
    tampered_lineage = bundle.raw[0].lineage.model_copy(
        update={"raw_object_hash": "sha256:" + "a" * 64}
    )
    tampered_parent = bundle.raw[0].model_copy(
        update={"lineage": tampered_lineage}
    )

    with pytest.raises(
        nfl.NFLModelInputError,
        match="state event envelope",
    ):
        nfl.fastrmodels_feature_vector(
            state,
            context=context,
            variant="spread",
            program_root=PROJECT_ROOT,
            state_event_raw_parents=(
                tampered_parent,
                bundle.raw[1],
            ),
        )


def test_raw_parent_times_must_equal_normalized_state_event_time() -> None:
    state, context, bundle = _fixture()
    changed_time = bundle.raw[0].time.model_dump(mode="python")
    changed_time["source_at"] = "2025-09-07T19:29:59Z"
    changed_parent = _recreate_envelope(
        bundle.raw[0],
        time=changed_time,
    )
    parents = (changed_parent, bundle.raw[1])
    normalized = _relink_normalized_parent_ids(
        bundle.normalized,
        parents,
    )
    state = replace(state, last_event_id=normalized.event_id)
    context = replace(context, state_event_envelope=normalized)

    with pytest.raises(nfl.NFLModelInputError, match="parent.*time"):
        nfl.fastrmodels_feature_vector(
            state,
            context=context,
            variant="spread",
            program_root=PROJECT_ROOT,
            state_event_raw_parents=parents,
        )


def test_context_source_object_must_equal_raw_parent_object_hash() -> None:
    state, context, bundle = _fixture()

    with pytest.raises(nfl.NFLModelInputError, match="source_object_sha256"):
        _feature_vector(
            state,
            replace(
                context,
                source_object_sha256="sha256:" + "a" * 64,
            ),
            bundle,
            variant="spread",
        )


def test_raw_parents_must_share_one_source_object_hash() -> None:
    state, context, bundle = _fixture()
    changed_hash = "sha256:" + "8" * 64
    changed_lineage = bundle.raw[0].lineage.model_dump(mode="python")
    changed_lineage["raw_object_hash"] = changed_hash
    changed_payload = dict(bundle.raw[0].payload)
    changed_payload["raw_object_hash"] = changed_hash
    changed_source = bundle.raw[0].source.model_dump(mode="python")
    changed_source["capture_session_id"] = f"static:{changed_hash}"
    changed_parent = _recreate_envelope(
        bundle.raw[0],
        lineage=changed_lineage,
        payload=changed_payload,
        source=changed_source,
    )
    parents = (changed_parent, bundle.raw[1])
    normalized = _relink_normalized_parent_ids(
        bundle.normalized,
        parents,
    )
    state = replace(state, last_event_id=normalized.event_id)
    context = replace(context, state_event_envelope=normalized)

    with pytest.raises(
        nfl.NFLModelInputError,
        match="single raw_object_hash",
    ):
        nfl.fastrmodels_feature_vector(
            state,
            context=context,
            variant="spread",
            program_root=PROJECT_ROOT,
            state_event_raw_parents=parents,
        )


def test_context_requires_a_contract_envelope() -> None:
    with pytest.raises(nfl.NFLModelInputError, match="state_event_envelope"):
        _context(state_event_envelope={"event_id": "not-validated"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "second_half_receiver_observed_at",
            datetime(2025, 9, 7, 17, 55),
        ),
        ("spread_observed_at", datetime(2025, 9, 7, 19, 0)),
        (
            "second_half_receiver_observed_at",
            datetime(
                2025,
                9,
                7,
                17,
                55,
                tzinfo=timezone(timedelta(hours=1)),
            ),
        ),
    ],
)
def test_external_observation_times_require_utc(
    field: str,
    value: datetime,
) -> None:
    _, context, _ = _fixture()
    with pytest.raises(nfl.NFLModelInputError, match=field):
        replace(context, **{field: value})
