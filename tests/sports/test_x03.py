from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pytest

from prediction_market.sports import x03


def _observations() -> list[dict[str, object]]:
    return [
        {
            "venue": "polymarket",
            "sport": "nba",
            "canonical_market_id": "market_nba_a",
            "observed_at": "2026-06-01T00:00:00Z",
            "best_bid": 0.40,
            "best_ask": 0.50,
            "depth": 100.0,
            "status": "trading",
            "pit_metadata_present": True,
        },
        {
            "venue": "polymarket",
            "sport": "nba",
            "canonical_market_id": "market_nba_a",
            "observed_at": "2026-06-01T01:00:00Z",
            "best_bid": 0.41,
            "best_ask": 0.51,
            "depth": 50.0,
            "status": "suspended",
            "pit_metadata_present": True,
        },
        {
            "venue": "polymarket",
            "sport": "nba",
            "canonical_market_id": "market_nba_b",
            "observed_at": "2026-06-01T02:00:00Z",
            "best_bid": 0.30,
            "best_ask": 0.90,
            "depth": 999.0,
            "status": "trading",
            "pit_metadata_present": False,
        },
        {
            "venue": "polymarket",
            "sport": "nba",
            "canonical_market_id": "market_nba_b",
            "observed_at": "2026-06-01T03:00:00Z",
            "best_bid": 0.30,
            "best_ask": 0.50,
            "depth": 80.0,
            "status": "trading",
            "pit_metadata_present": True,
        },
        {
            "venue": "kalshi",
            "sport": "nfl",
            "canonical_market_id": "market_nfl_a",
            "observed_at": "2026-06-02T00:00:00Z",
            "best_bid": 0.55,
            "best_ask": 0.65,
            "depth": 30.0,
            "status": "trading",
            "pit_metadata_present": True,
        },
    ]


def _document(
    *,
    observations: list[dict[str, object]] | None = None,
    start_at: str = "2026-06-01T00:00:00Z",
    end_at: str = "2026-06-29T00:00:00Z",
) -> dict[str, object]:
    document: dict[str, object] = {
        "document_version": "x03-normalized-input/v0",
        "experiment_id": "X-03",
        "manifest_sha256": "sha256:" + "a" * 64,
        "window": {
            "start_at": start_at,
            "end_at": end_at,
        },
        "sample_interval_seconds": 3600,
        "observations": observations if observations is not None else _observations(),
    }
    document["input_sha256"] = x03.normalized_input_sha256(document)
    return document


def _payload(document: dict[str, object]) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_census_is_deterministic_and_excludes_missing_pit_from_liquidity() -> None:
    original = _document()
    reordered = _document(observations=list(reversed(_observations())))

    first = x03.run_x03_census(_payload(original))
    second = x03.run_x03_census(_payload(reordered))

    assert first == second
    assert first["status"] == "POC_ONLY"
    assert first["is_formal_result"] is False
    assert first["formal_result_eligible"] is False
    assert "results_ref" not in first
    assert "promotion_decision" not in first
    assert first["input"]["window"] == {
        "start_at": "2026-06-01T00:00:00Z",
        "end_at": "2026-06-29T00:00:00Z",
        "duration_days": 28,
    }
    assert first["input"]["normalized_input_sha256"] == original["input_sha256"]
    assert first["totals"] == {
        "observation_count": 5,
        "market_count": 3,
        "missing_pit_metadata_count": 1,
    }
    assert first["sport_venue_census"] == [
        {
            "sport": "nba",
            "venue": "polymarket",
            "market_count": 2,
            "in_play_hours": 2.0,
            "median_executable_spread": 0.15,
            "median_depth": 90.0,
            "pause_frequency": 1 / 3,
            "eligible_observation_count": 3,
            "trading_observation_count": 2,
            "suspended_observation_count": 1,
            "missing_pit_metadata_count": 1,
        },
        {
            "sport": "nfl",
            "venue": "kalshi",
            "market_count": 1,
            "in_play_hours": 1.0,
            "median_executable_spread": 0.10,
            "median_depth": 30.0,
            "pause_frequency": 0.0,
            "eligible_observation_count": 1,
            "trading_observation_count": 1,
            "suspended_observation_count": 0,
            "missing_pit_metadata_count": 0,
        },
    ]
    assert first["result_sha256"] == x03.result_sha256(first)


def test_suspended_only_group_has_no_executable_liquidity_medians() -> None:
    observation = _observations()[1]
    result = x03.run_x03_census(_payload(_document(observations=[observation])))
    group = result["sport_venue_census"][0]

    assert group["market_count"] == 1
    assert group["in_play_hours"] == 0.0
    assert group["median_executable_spread"] is None
    assert group["median_depth"] is None
    assert group["pause_frequency"] == 1.0


def test_duplicate_json_key_is_rejected() -> None:
    payload = _payload(_document())
    payload = payload.replace(
        b'"experiment_id":"X-03",',
        b'"experiment_id":"X-03","experiment_id":"X-03",',
        1,
    )

    with pytest.raises(x03.X03DataError, match="duplicate JSON key"):
        x03.run_x03_census(payload)


@pytest.mark.parametrize("location", ["top", "window", "observation"])
def test_unknown_fields_are_rejected(location: str) -> None:
    document = _document()
    if location == "top":
        document["unexpected"] = True
    elif location == "window":
        window = document["window"]
        assert isinstance(window, dict)
        window["unexpected"] = True
    else:
        observations = document["observations"]
        assert isinstance(observations, list)
        observations[0]["unexpected"] = True

    with pytest.raises(x03.X03DataError, match="fields"):
        x03.run_x03_census(_payload(document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_at", "2026-06-01T00:00:00+00:00"),
        ("start_at", "2026-6-01T00:00:00Z"),
        ("end_at", "2026-06-29T00:00:00.000Z"),
    ],
)
def test_noncanonical_window_timestamps_are_rejected(
    field: str, value: str
) -> None:
    document = _document()
    window = document["window"]
    assert isinstance(window, dict)
    window[field] = value

    with pytest.raises(x03.X03DataError, match="canonical UTC"):
        x03.run_x03_census(_payload(document))


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-06-01T00:00:00+00:00",
        "2026-06-01T00:00:00.000Z",
    ],
)
def test_noncanonical_observation_timestamp_is_rejected(timestamp: str) -> None:
    document = _document()
    observations = document["observations"]
    assert isinstance(observations, list)
    observations[0]["observed_at"] = timestamp

    with pytest.raises(x03.X03DataError, match="canonical UTC"):
        x03.run_x03_census(_payload(document))


def test_window_must_be_exactly_28_days() -> None:
    document = _document()
    window = document["window"]
    assert isinstance(window, dict)
    window["end_at"] = "2026-06-28T00:00:00Z"

    with pytest.raises(x03.X03DataError, match="exactly 28 days"):
        x03.run_x03_census(_payload(document))


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-05-31T23:00:00Z",
        "2026-06-29T00:00:00Z",
        "2026-06-28T23:30:01Z",
    ],
)
def test_mixed_or_out_of_window_observations_are_rejected(
    timestamp: str,
) -> None:
    document = _document()
    observations = document["observations"]
    assert isinstance(observations, list)
    observations[0]["observed_at"] = timestamp

    with pytest.raises(x03.X03DataError, match="28-day window"):
        x03.run_x03_census(_payload(document))


def test_observation_must_follow_the_locked_hourly_sample_grid() -> None:
    document = _document()
    observations = document["observations"]
    assert isinstance(observations, list)
    observations[0]["observed_at"] = "2026-06-01T00:30:00Z"

    with pytest.raises(x03.X03DataError, match="sample grid"):
        x03.run_x03_census(_payload(document))


def test_crossed_executable_quote_is_rejected() -> None:
    document = _document()
    observations = document["observations"]
    assert isinstance(observations, list)
    observations[0]["best_bid"] = 0.7
    observations[0]["best_ask"] = 0.6

    with pytest.raises(x03.X03DataError, match="ask"):
        x03.run_x03_census(_payload(document))


def test_negative_depth_is_rejected() -> None:
    document = _document()
    observations = document["observations"]
    assert isinstance(observations, list)
    observations[0]["depth"] = -1

    with pytest.raises(x03.X03DataError, match="depth"):
        x03.run_x03_census(_payload(document))


def test_tiny_negative_depth_cannot_round_to_accepted_negative_zero() -> None:
    payload = _payload(_document()).replace(
        b'"depth":100.0',
        b'"depth":-1e-400',
        1,
    )

    with pytest.raises(x03.X03DataError, match="depth must be non-negative"):
        x03.run_x03_census(payload)


def test_high_precision_crossed_quote_cannot_alias_to_equal_floats() -> None:
    payload = _payload(_document()).replace(
        b'"best_ask":0.5,"best_bid":0.4',
        (
            b'"best_ask":0.5,'
            b'"best_bid":0.5000000000000000000000000000000001'
        ),
        1,
    )

    with pytest.raises(x03.X03DataError, match="best_ask cannot be below best_bid"):
        x03.run_x03_census(payload)


def test_supported_exact_numeric_values_remain_distinct_in_input_hash() -> None:
    lower = _document()
    lower_observations = lower["observations"]
    assert isinstance(lower_observations, list)
    lower_observations[0]["depth"] = 0.1
    lower["input_sha256"] = x03.normalized_input_sha256(lower)

    upper = _document()
    upper_observations = upper["observations"]
    assert isinstance(upper_observations, list)
    upper_observations[0]["depth"] = 0.10000000000000002
    upper["input_sha256"] = x03.normalized_input_sha256(upper)

    assert lower["input_sha256"] != upper["input_sha256"]
    assert (
        x03.run_x03_census(_payload(lower))["input"]["normalized_input_sha256"]
        == lower["input_sha256"]
    )
    assert (
        x03.run_x03_census(_payload(upper))["input"]["normalized_input_sha256"]
        == upper["input_sha256"]
    )


def test_unsupported_decimal_precision_is_rejected_before_hashing() -> None:
    payload = _payload(_document()).replace(
        b'"depth":100.0',
        b'"depth":0.10000000000000001',
        1,
    )

    with pytest.raises(x03.X03DataError, match="unsupported decimal precision"):
        x03.run_x03_census(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("venue", "unknown"),
        ("sport", "quidditch"),
        ("status", "closed"),
    ],
)
def test_unknown_venue_sport_or_status_is_rejected(
    field: str, value: str
) -> None:
    document = _document()
    observations = document["observations"]
    assert isinstance(observations, list)
    observations[0][field] = value

    with pytest.raises(x03.X03DataError, match=field):
        x03.run_x03_census(_payload(document))


@pytest.mark.parametrize("field", ["manifest_sha256", "input_sha256"])
def test_missing_manifest_or_input_hash_is_rejected(field: str) -> None:
    document = _document()
    del document[field]

    with pytest.raises(x03.X03DataError, match="fields"):
        x03.run_x03_census(_payload(document))


def test_invalid_manifest_digest_is_rejected() -> None:
    document = _document()
    document["manifest_sha256"] = "sha256:not-a-digest"

    with pytest.raises(x03.X03DataError, match="manifest_sha256"):
        x03.run_x03_census(_payload(document))


def test_input_hash_tamper_is_rejected() -> None:
    document = _document()
    observations = document["observations"]
    assert isinstance(observations, list)
    observations[0]["depth"] = 101.0

    with pytest.raises(x03.X03DataError, match="input_sha256"):
        x03.run_x03_census(_payload(document))


def test_duplicate_logical_observation_is_rejected() -> None:
    document = _document()
    observations = document["observations"]
    assert isinstance(observations, list)
    observations.append(copy.deepcopy(observations[0]))

    with pytest.raises(x03.X03DataError, match="duplicate observation"):
        x03.run_x03_census(_payload(document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pit_metadata_present", 1),
        ("best_bid", True),
        ("depth", "100"),
    ],
)
def test_observation_field_types_fail_closed(
    field: str, value: object
) -> None:
    document = _document()
    observations = document["observations"]
    assert isinstance(observations, list)
    observations[0][field] = value

    with pytest.raises(x03.X03DataError, match=field):
        x03.run_x03_census(_payload(document))


def test_result_self_hash_detects_mutation() -> None:
    result = x03.run_x03_census(_payload(_document()))
    mutated = copy.deepcopy(result)
    mutated["totals"]["observation_count"] = 999

    assert x03.result_sha256(mutated) != result["result_sha256"]


def test_nonfinite_json_is_rejected() -> None:
    document = _document()
    payload = _payload(document).replace(b'"depth":100.0', b'"depth":NaN', 1)

    with pytest.raises(x03.X03DataError, match="non-finite"):
        x03.run_x03_census(payload)


def test_input_is_bytes_and_sample_interval_is_one_hour() -> None:
    document = _document()
    document["sample_interval_seconds"] = 60

    with pytest.raises(x03.X03DataError, match="sample_interval_seconds"):
        x03.run_x03_census(_payload(document))
    with pytest.raises(TypeError, match="bytes"):
        x03.run_x03_census(json.loads(_payload(_document())))  # type: ignore[arg-type]


def test_window_endpoint_parser_rejects_invalid_calendar_time() -> None:
    document = _document()
    window = document["window"]
    assert isinstance(window, dict)
    window["start_at"] = datetime(
        2026, 6, 1, tzinfo=timezone.utc
    ).strftime("%Y-%m-%dT25:00:00Z")

    with pytest.raises(x03.X03DataError, match="canonical UTC"):
        x03.run_x03_census(_payload(document))
