from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import prediction_market.sports.nfl_x13_batch as batch_module
from prediction_market.sports import nfl_x13_dashboard as dashboard
from prediction_market.sports.nfl_x13_spec import X13_GAME_IDS


_BATCH_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_nfl_x13_batch_fixtures",
    Path(__file__).with_name("test_nfl_x13_batch.py"),
)
assert _BATCH_FIXTURE_SPEC is not None
assert _BATCH_FIXTURE_SPEC.loader is not None
batch_fixtures = importlib.util.module_from_spec(_BATCH_FIXTURE_SPEC)
sys.modules[_BATCH_FIXTURE_SPEC.name] = batch_fixtures
_BATCH_FIXTURE_SPEC.loader.exec_module(batch_fixtures)


BATCH_ID = (
    "sha256:"
    "1111111111111111111111111111111111111111111111111111111111111111"
)
SCRIPT_OPEN = '<script id="x13-data" type="application/json">'


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _presentation(game_id: str) -> dict[str, object]:
    logical_market_id = f"kalshi:{game_id}:winner"
    return {
        "schema": "nfl_x13_html_presentation_v1",
        "experiment_id": "X-13",
        "status": "PRELIMINARY_SOURCE_TIME_ONLY",
        "game_id": game_id,
        "teams": {
            "away": "AAA",
            "home": "BBB",
            "away_color": "#123456",
            "home_color": "#abcdef",
        },
        "final_score": {"away": 17, "home": 20},
        "events": [{"play_id": f"{game_id}:1", "description": "Run"}],
        "episodes": [
            {
                "episode_id": f"{game_id}:episode:1",
                "episode_type": "routine",
            }
        ],
        "stat_ledger": [{"through_play_id": f"{game_id}:1"}],
        "personnel": {"coverage": "research_only"},
        "contracts": [
            {
                "logical_market_id": logical_market_id,
                "contract_id": f"{game_id}:contract",
                "venue": "kalshi",
                "family": "moneyline",
                "outcomes": ["AAA", "BBB"],
                "outcome_coverage": {"AAA": "available", "BBB": "available"},
                "selector_key": logical_market_id,
                "selector_label": "AAA vs BBB",
            }
        ],
        "market_series": [
            {
                "series_kind": "trade_range",
                "logical_market_id": logical_market_id,
                "contract_id": f"{game_id}:contract",
                "venue": "kalshi",
                "family": "moneyline",
                "outcome": "AAA",
                "provenance": "observed",
                "points": [
                    {
                        "source_start": "2025-09-01T00:00:00Z",
                        "source_end": "2025-09-01T00:00:01Z",
                        "price_low": "0.40",
                        "price_high": "0.42",
                    }
                ],
            }
        ],
        "association_preview": [
            {
                "episode_id": f"{game_id}:episode:1",
                "logical_market_id": logical_market_id,
                "delay_scenario_seconds": 0,
                "horizon_seconds": 5,
                "order_ambiguous": False,
                "contaminated": False,
            }
        ],
        "association_coverage": {
            "full_row_count": 100,
            "preview_row_count": 1,
            "full_table_partitions": [],
        },
        "observation_coverage": {
            "full_observation_count": 1,
            "embedded_series_count": 1,
            "embedded_point_count": 1,
            "point_limit_per_series": 512,
        },
        "presentation_limits": {
            "point_limit_per_series": 512,
            "association_dom_page_size": 100,
        },
        "market_coverage": {
            "kalshi_historical_bbo": "available",
            "polymarket_historical_bbo": "unavailable",
        },
        "audit": {"layer_g": "PASS", "layer_m": "PASS", "layer_a": "PASS"},
        "lineage": {"builder_version": "nfl-x13-pipeline-v2"},
    }


def _write_batch(tmp_path: Path) -> Path:
    root = tmp_path / "batch"
    games = root / "games"
    games.mkdir(parents=True)
    checksums: dict[str, str] = {}
    for game_id in X13_GAME_IDS:
        embedded = json.dumps(
            _presentation(game_id),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).replace("<", "\\u003c")
        html = (
            "<!doctype html><html><body>"
            f"{SCRIPT_OPEN}{embedded}</script>"
            "</body></html>"
        ).encode("utf-8")
        relative = f"games/{game_id}.html"
        (root / relative).write_bytes(html)
        checksums[relative] = hashlib.sha256(html).hexdigest()
    manifest = {
        "schema": "nfl_x13_batch_manifest_v1",
        "batch_id": BATCH_ID,
        "experiment_id": "X-13",
        "game_ids": list(X13_GAME_IDS),
        "game_count": 20,
        "publication_gate": "PASS",
        "builder_version": "nfl-x13-pipeline-v2",
    }
    manifest_bytes = _canonical_bytes(manifest)
    (root / "batch_manifest.json").write_bytes(manifest_bytes)
    checksums["batch_manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
    ledger = "".join(
        f"{digest}  {relative}\n"
        for relative, digest in sorted(checksums.items())
    )
    (root / "checksums.sha256").write_text(ledger, encoding="ascii")
    return root


def _replace_game_presentation(
    batch: Path,
    game_id: str,
    presentation: dict[str, object],
    *,
    script_open: str = SCRIPT_OPEN,
) -> None:
    embedded = json.dumps(
        presentation,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).replace("<", "\\u003c")
    html = (
        "<!doctype html><html><body>"
        f"{script_open}{embedded}</script>"
        "</body></html>"
    ).encode("utf-8")
    relative = f"games/{game_id}.html"
    (batch / relative).write_bytes(html)
    entries: dict[str, str] = {}
    for line in (batch / "checksums.sha256").read_text("ascii").splitlines():
        digest, path = line.split("  ", 1)
        entries[path] = digest
    entries[relative] = hashlib.sha256(html).hexdigest()
    (batch / "checksums.sha256").write_text(
        "".join(
            f"{digest}  {path}\n" for path, digest in sorted(entries.items())
        ),
        encoding="ascii",
    )


def _stub_batch_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "verify_published_batch",
        lambda _path: {
            "batch_id": BATCH_ID,
            "game_count": len(X13_GAME_IDS),
        },
    )


def _write_real_verified_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    payloads = batch_fixtures._payloads()
    monkeypatch.setattr(
        batch_module,
        "X13_FROZEN_ASSOCIATION_ROWS_BY_GAME",
        {game_id: 144 for game_id in X13_GAME_IDS},
    )
    monkeypatch.setattr(
        batch_module,
        "X13_FROZEN_ASSOCIATION_ROW_COUNT",
        144 * len(X13_GAME_IDS),
    )
    monkeypatch.setattr(
        batch_module,
        "X13_FROZEN_CONTRACT_INVENTORY_SHA256_BY_GAME",
        {
            str(payload["game_id"]): (
                batch_module.contract_inventory_sha256(
                    payload["contracts"]
                )
            )
            for payload in payloads
        },
    )
    return batch_fixtures._publish_batch(
        payloads, output_root=tmp_path / "real-batch"
    )


def test_extract_presentation_requires_exact_schema_and_source_hash(
    tmp_path: Path,
) -> None:
    game_id = X13_GAME_IDS[0]
    value = _presentation(game_id)
    embedded = json.dumps(value, separators=(",", ":"), sort_keys=True)
    html = f"{SCRIPT_OPEN}{embedded}</script>".encode()
    path = tmp_path / "game.html"
    path.write_bytes(html)

    extracted = dashboard.extract_presentation_html(
        path,
        expected_sha256=f"sha256:{hashlib.sha256(html).hexdigest()}",
        expected_game_id=game_id,
    )

    assert extracted == value
    value["schema"] = "foreign"
    foreign = f"{SCRIPT_OPEN}{json.dumps(value)}</script>".encode()
    path.write_bytes(foreign)
    with pytest.raises(dashboard.X13DashboardExportError, match="schema"):
        dashboard.extract_presentation_html(
            path,
            expected_sha256=f"sha256:{hashlib.sha256(foreign).hexdigest()}",
            expected_game_id=game_id,
        )


def test_extract_accepts_attribute_variations_but_rejects_duplicate_id(
    tmp_path: Path,
) -> None:
    game_id = X13_GAME_IDS[0]
    embedded = json.dumps(_presentation(game_id), separators=(",", ":"))
    varied_open = (
        "<script data-projection='x13' type = 'application/json' "
        "id = 'x13-data'>"
    )
    html = f"{varied_open}{embedded}</script>".encode()
    path = tmp_path / "game.html"
    path.write_bytes(html)

    assert dashboard.extract_presentation_html(
        path,
        expected_sha256=f"sha256:{hashlib.sha256(html).hexdigest()}",
        expected_game_id=game_id,
    )["game_id"] == game_id

    duplicate = (
        f"{varied_open}{embedded}</script>"
        f"{SCRIPT_OPEN}{embedded}</script>"
    ).encode()
    path.write_bytes(duplicate)
    with pytest.raises(
        dashboard.X13DashboardExportError, match="exactly one|duplicate"
    ):
        dashboard.extract_presentation_html(
            path,
            expected_sha256=(
                f"sha256:{hashlib.sha256(duplicate).hexdigest()}"
            ),
            expected_game_id=game_id,
        )


def test_extract_rejects_missing_script(tmp_path: Path) -> None:
    path = tmp_path / "game.html"
    raw = b"<html><body>no embedded presentation</body></html>"
    path.write_bytes(raw)

    with pytest.raises(
        dashboard.X13DashboardExportError, match="exactly one|x13-data"
    ):
        dashboard.extract_presentation_html(
            path,
            expected_sha256=f"sha256:{hashlib.sha256(raw).hexdigest()}",
            expected_game_id=X13_GAME_IDS[0],
        )


def test_export_is_deterministic_bounded_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)

    first = dashboard.export_x13_dashboard(batch, tmp_path / "first")
    second = dashboard.export_x13_dashboard(batch, tmp_path / "second")

    assert isinstance(first, dashboard.DashboardExportV1)
    assert first.game_count == 20
    assert first.maximum_asset_bytes <= dashboard.MAX_ASSET_BYTES
    assert first.maximum_asset_bytes == max(
        first.manifest_path.stat().st_size,
        *(asset.byte_length for asset in first.assets),
    )
    assert all(
        isinstance(asset, dashboard.DashboardAssetV1)
        for asset in first.assets
    )
    assert {field.name for field in fields(dashboard.DashboardAssetV1)} >= {
        "role",
        "source_sha256",
    }
    assert "logical_role" not in {
        field.name for field in fields(dashboard.DashboardAssetV1)
    }
    assert "publish_manifest_sha256" in {
        field.name for field in fields(dashboard.DashboardExportV1)
    }
    assert "__dict__" not in dashboard.DashboardAssetV1.__slots__
    assert "__dict__" not in dashboard.DashboardExportV1.__slots__
    with pytest.raises(FrozenInstanceError):
        first.assets[0].role = "changed"
    assert all(asset.source_sha256.startswith("sha256:") for asset in first.assets)
    assert (
        first.publish_manifest_sha256
        == second.publish_manifest_sha256
    )
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    manifest = json.loads(first.manifest_path.read_bytes())
    assert manifest["schema"] == "nfl_x13_dashboard_publish_manifest_v1"
    assert manifest["batch_id"] == BATCH_ID
    assert manifest["game_count"] == 20
    assert manifest["asset_count"] == len(manifest["assets"])
    roles = [row["logical_role"] for row in manifest["assets"]]
    assert roles.count("catalog") == 1
    assert roles.count("game_core") == 20
    assert roles.count("game_contracts") == 20
    assert roles.count("association_preview") == 20
    assert roles.count("market_series") == 20
    for descriptor in manifest["assets"]:
        relative = Path(descriptor["relative_path"])
        path = first.publish_root / relative
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        assert len(raw) <= dashboard.MAX_ASSET_BYTES
        assert descriptor["byte_length"] == len(raw)
        assert descriptor["object_sha256"] == (
            f"sha256:{hashlib.sha256(raw).hexdigest()}"
        )
        assert relative.name == (
            descriptor["object_sha256"].replace(":", "-") + ".json"
        )
        assert descriptor["source_artifacts"]
        assert descriptor["source_sha256"].startswith("sha256:")
        json.loads(raw)


def test_public_projection_verifier_reopens_exact_task2_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    exported = dashboard.export_x13_dashboard(
        batch, tmp_path / "output"
    )

    verified = dashboard.verify_x13_dashboard_export(
        exported.publish_root
    )

    assert verified.publish_root == exported.publish_root
    assert (
        verified.manifest_sha256
        == exported.publish_manifest_sha256
    )
    assert verified.batch_id == exported.batch_id
    assert len(verified.assets) == exported.asset_count
    assert all(asset.payload for asset in verified.assets)


def test_public_projection_verifier_rejects_forged_embedded_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    exported = dashboard.export_x13_dashboard(
        batch, tmp_path / "output"
    )
    manifest_path = exported.manifest_path
    manifest = json.loads(manifest_path.read_bytes())
    descriptor = manifest["assets"][0]
    old_asset = exported.publish_root / descriptor["relative_path"]
    document = json.loads(old_asset.read_bytes())
    document["builder_version"] = "forged-builder"
    replacement_raw = _canonical_bytes(document)
    replacement_sha = (
        "sha256:" + hashlib.sha256(replacement_raw).hexdigest()
    )
    replacement_relative = (
        old_asset.parent.relative_to(exported.publish_root)
        / f"{replacement_sha.replace(':', '-')}.json"
    ).as_posix()
    old_asset.unlink()
    (exported.publish_root / replacement_relative).write_bytes(
        replacement_raw
    )
    descriptor["relative_path"] = replacement_relative
    descriptor["object_sha256"] = replacement_sha
    descriptor["byte_length"] = len(replacement_raw)
    manifest["assets"].sort(key=lambda row: row["relative_path"])
    replacement_manifest_raw = _canonical_bytes(manifest)
    replacement_manifest_sha = (
        "sha256:" + hashlib.sha256(replacement_manifest_raw).hexdigest()
    )
    manifest_path.unlink()
    (
        manifest_path.parent
        / f"{replacement_manifest_sha.replace(':', '-')}.json"
    ).write_bytes(replacement_manifest_raw)

    with pytest.raises(
        dashboard.X13DashboardExportError,
        match="embedded.*identity|builder",
    ):
        dashboard.verify_x13_dashboard_export(
            exported.publish_root
        )


def test_export_rejects_tampered_source_after_batch_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    source = batch / "games" / f"{X13_GAME_IDS[0]}.html"
    source.write_bytes(source.read_bytes() + b" ")

    with pytest.raises(dashboard.X13DashboardExportError, match="source hash"):
        dashboard.export_x13_dashboard(batch, tmp_path / "output")


def test_batch_reads_remain_bound_to_verified_open_directory_after_path_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path / "original")
    replacement = _write_batch(tmp_path / "replacement")
    replacement_value = _presentation(X13_GAME_IDS[0])
    replacement_value["events"] = [
        {"play_id": "foreign", "description": "REPLACEMENT PATH"}
    ]
    _replace_game_presentation(
        replacement, X13_GAME_IDS[0], replacement_value
    )
    moved = tmp_path / "verified-inode"

    def replace_path_after_verify(path: Path) -> dict[str, object]:
        assert path != batch
        assert path.name == "batch"
        batch.rename(moved)
        replacement.rename(batch)
        return {"batch_id": BATCH_ID, "game_count": len(X13_GAME_IDS)}

    monkeypatch.setattr(
        dashboard, "verify_published_batch", replace_path_after_verify
    )

    result = dashboard.export_x13_dashboard(batch, tmp_path / "output")
    core = next(
        asset
        for asset in result.assets
        if asset.game_id == X13_GAME_IDS[0] and asset.role == "game_core"
    )
    document = json.loads((result.publish_root / core.relative_path).read_bytes())
    assert document["payload"]["events"][0]["description"] == "Run"


@pytest.mark.parametrize("mutation", ("path_swap", "in_place_source_replace"))
def test_real_full_verifier_and_projection_use_same_private_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    batch = _write_real_verified_batch(tmp_path, monkeypatch)
    full_verifier = dashboard.verify_published_batch
    original_html = batch / "games" / f"{X13_GAME_IDS[0]}.html"
    original_bytes = original_html.read_bytes()
    moved = tmp_path / "moved-real-batch"

    def mutate_original_then_full_verify(
        snapshot_path: Path,
    ) -> dict[str, object]:
        assert snapshot_path != batch
        if mutation == "path_swap":
            batch.rename(moved)
            batch.mkdir()
            (batch / "foreign").write_text("not the verified batch")
        else:
            original_html.write_bytes(b"foreign replacement")
        return full_verifier(snapshot_path)

    monkeypatch.setattr(
        dashboard,
        "verify_published_batch",
        mutate_original_then_full_verify,
    )

    result = dashboard.export_x13_dashboard(batch, tmp_path / "output")
    source_sha = f"sha256:{hashlib.sha256(original_bytes).hexdigest()}"
    game_assets = [
        asset
        for asset in result.assets
        if asset.game_id == X13_GAME_IDS[0]
    ]
    assert game_assets
    assert all(asset.source_sha256 == source_sha for asset in game_assets)


def test_html_size_limit_fails_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "oversize.html"
    raw = b"\xff" * 129
    path.write_bytes(raw)
    monkeypatch.setattr(dashboard, "MAX_SOURCE_HTML_BYTES", 128)

    with pytest.raises(dashboard.X13DashboardExportError, match="size limit"):
        dashboard.extract_presentation_html(
            path,
            expected_sha256=f"sha256:{hashlib.sha256(raw).hexdigest()}",
            expected_game_id=X13_GAME_IDS[0],
        )


def test_snapshot_destination_open_failure_closes_source_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "object").write_bytes(b"source")
    source_parent_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
    destination_parent_fd = os.open(
        destination, os.O_RDONLY | os.O_DIRECTORY
    )
    real_open = os.open
    real_close = os.close
    opened_source: list[int] = []
    closed: list[int] = []

    def fail_destination_open(
        path: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        if path == "object" and flags & os.O_WRONLY:
            raise OSError("injected destination-open failure")
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == "object" and not flags & os.O_WRONLY:
            opened_source.append(descriptor)
        return descriptor

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(dashboard.os, "open", fail_destination_open)
    monkeypatch.setattr(dashboard.os, "close", record_close)
    try:
        with pytest.raises(OSError, match="injected"):
            dashboard._copy_snapshot_file(
                source_parent_fd,
                destination_parent_fd,
                "object",
                remaining_total=1024,
            )
        assert opened_source
        assert opened_source[0] in closed
    finally:
        real_close(source_parent_fd)
        real_close(destination_parent_fd)


def test_snapshot_setup_failure_closes_open_source_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    opened: list[int] = []
    real_open_directory = dashboard._open_existing_directory
    real_mkdir = Path.mkdir

    def record_open(path: Path) -> tuple[int, tuple[int, int]]:
        result = real_open_directory(path)
        opened.append(result[0])
        return result

    def fail_snapshot_mkdir(
        path: Path, *args: object, **kwargs: object
    ) -> None:
        if ".x13-dashboard-batch-snapshot-" in str(path.parent):
            raise OSError("injected snapshot mkdir failure")
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(dashboard, "_open_existing_directory", record_open)
    monkeypatch.setattr(Path, "mkdir", fail_snapshot_mkdir)

    with pytest.raises(OSError, match="injected"):
        dashboard.export_x13_dashboard(batch, tmp_path / "output")
    assert opened
    for descriptor in opened:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_export_rejects_duplicate_logical_market(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    game_id = X13_GAME_IDS[0]
    presentation = _presentation(game_id)
    contracts = list(presentation["contracts"])
    contracts.append(dict(contracts[0]))
    presentation["contracts"] = contracts
    _replace_game_presentation(batch, game_id, presentation)

    with pytest.raises(
        dashboard.X13DashboardExportError, match="duplicate logical market"
    ):
        dashboard.export_x13_dashboard(batch, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "unsafe"),
    (("venue", "../outside"), ("family", "moneyline/../../outside")),
)
def test_export_rejects_unsafe_market_path_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    unsafe: str,
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    game_id = X13_GAME_IDS[0]
    presentation = _presentation(game_id)
    contracts = list(presentation["contracts"])
    contract = dict(contracts[0])
    contract[field] = unsafe
    contracts[0] = contract
    presentation["contracts"] = contracts
    _replace_game_presentation(batch, game_id, presentation)

    with pytest.raises(
        dashboard.X13DashboardExportError, match="safe path component"
    ):
        dashboard.export_x13_dashboard(batch, tmp_path / "output")
    assert not (tmp_path / "outside").exists()


def _polymarket_presentation(game_id: str) -> dict[str, object]:
    presentation = _presentation(game_id)
    logical_market_id = f"polymarket:{game_id}:winner"
    presentation["contracts"] = [
        {
            "logical_market_id": logical_market_id,
            "contract_id": f"{game_id}:poly-contract",
            "venue": "polymarket",
            "family": "moneyline",
            "outcomes": ["AAA", "BBB"],
            "outcome_coverage": {
                "AAA": "available",
                "BBB": "derived_non_observed",
            },
            "selector_key": logical_market_id,
            "selector_label": "AAA vs BBB",
        }
    ]
    base = {
        "series_kind": "trade_range",
        "logical_market_id": logical_market_id,
        "contract_id": f"{game_id}:poly-contract",
        "venue": "polymarket",
        "family": "moneyline",
        "points": [{"source_start": "2025-09-01T00:00:00Z", "price": "0.4"}],
    }
    presentation["market_series"] = [
        {**base, "outcome": "AAA", "provenance": "observed"},
        {
            **base,
            "outcome": "BBB",
            "provenance": "derived_complement",
            "points": [
                {"source_start": "2025-09-01T00:00:00Z", "price": "0.6"}
            ],
        },
    ]
    presentation["market_coverage"] = {
        "kalshi_historical_bbo": "available_candle_bbo",
        "polymarket_historical_bbo": "unavailable",
        "robinhood_historical_venue": "provenance_only",
    }
    return presentation


def test_export_preserves_observed_and_derived_without_inventing_poly_bbo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    game_id = X13_GAME_IDS[0]
    presentation = _polymarket_presentation(game_id)
    _replace_game_presentation(batch, game_id, presentation)

    result = dashboard.export_x13_dashboard(batch, tmp_path / "output")
    market = next(
        asset
        for asset in result.assets
        if asset.game_id == game_id and asset.role == "market_series"
    )
    document = json.loads((result.publish_root / market.relative_path).read_bytes())
    series = document["payload"]["series"]
    assert {row["provenance"] for row in series} == {
        "observed",
        "derived_complement",
    }
    assert {row["series_kind"] for row in series} == {"trade_range"}
    assert all("bid" not in point and "ask" not in point for row in series for point in row["points"])


def test_export_rejects_invented_polymarket_bbo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    game_id = X13_GAME_IDS[0]
    presentation = _polymarket_presentation(game_id)
    series = list(presentation["market_series"])
    invented = dict(series[0])
    invented["series_kind"] = "bbo_candle"
    invented["points"] = [
        {
            "source_start": "2025-09-01T00:00:00Z",
            "bid_open": "0.39",
            "ask_open": "0.41",
        }
    ]
    series.append(invented)
    presentation["market_series"] = series
    _replace_game_presentation(batch, game_id, presentation)

    with pytest.raises(
        dashboard.X13DashboardExportError,
        match="Polymarket.*BBO|invented",
    ):
        dashboard.export_x13_dashboard(batch, tmp_path / "output")


def test_export_rejects_symlinked_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(dashboard.X13DashboardExportError, match="symlink"):
        dashboard.export_x13_dashboard(batch, alias)
    assert list(outside.iterdir()) == []


def test_atomic_publish_never_replaces_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    output = tmp_path / "output"
    output.mkdir()
    target = output / f"batch={BATCH_ID.replace(':', '-')}"
    target.mkdir()
    marker = target / "user-owned"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(dashboard.X13DashboardExportError, match="exists"):
        dashboard.export_x13_dashboard(batch, output)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_publish_fails_closed_when_asset_is_tampered_during_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    original = dashboard._atomic_rename_noreplace

    def publish_then_tamper(
        output_fd: int,
        source_name: str,
        target_name: str,
        **kwargs: object,
    ) -> None:
        original(output_fd, source_name, target_name, **kwargs)
        target = tmp_path / "output" / target_name
        asset = next(
            path
            for path in target.rglob("*.json")
            if path.parent.name != "manifests"
        )
        asset.write_bytes(b'{"tampered":true}\n')

    monkeypatch.setattr(
        dashboard, "_atomic_rename_noreplace", publish_then_tamper
    )

    with pytest.raises(
        dashboard.X13DashboardExportError,
        match=(
            "published but verification failed.*manual inspection required"
        ),
    ):
        dashboard.export_x13_dashboard(batch, tmp_path / "output")
    target = (
        tmp_path / "output" / f"batch={BATCH_ID.replace(':', '-')}"
    )
    assert target.is_dir()
    assert any(
        path.read_bytes() == b'{"tampered":true}\n'
        for path in target.rglob("*.json")
    )
    monkeypatch.setattr(dashboard, "_atomic_rename_noreplace", original)
    with pytest.raises(
        dashboard.X13DashboardExportError, match="target already exists"
    ):
        dashboard.export_x13_dashboard(batch, tmp_path / "output")


def test_output_path_redirect_race_never_writes_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    outside.mkdir()
    anchored = tmp_path / "anchored-output"
    original = dashboard._atomic_rename_noreplace
    redirected = False

    def redirect_then_publish(
        output_fd: int,
        source_name: str,
        target_name: str,
        **kwargs: object,
    ) -> None:
        nonlocal redirected
        if not redirected:
            output.rename(anchored)
            output.symlink_to(outside, target_is_directory=True)
            redirected = True
        original(output_fd, source_name, target_name, **kwargs)

    monkeypatch.setattr(
        dashboard, "_atomic_rename_noreplace", redirect_then_publish
    )

    with pytest.raises(
        dashboard.X13DashboardExportError,
        match=(
            "published but verification failed.*manual inspection required"
        ),
    ):
        dashboard.export_x13_dashboard(batch, output)
    assert list(outside.iterdir()) == []
    assert (
        anchored / f"batch={BATCH_ID.replace(':', '-')}"
    ).is_dir()


def test_post_rename_foreign_replacement_is_never_mutated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    output = tmp_path / "output"
    production_atomic = dashboard._atomic_rename_noreplace
    calls = {"atomic": 0, "rename": 0, "unlink": 0, "rmdir": 0}
    foreign_anchor: tuple[int, int] | None = None
    marker_bytes = b"foreign target must remain byte-exact"
    real_rename = os.rename
    real_unlink = os.unlink
    real_rmdir = os.rmdir

    def publish_then_replace(
        output_fd: int,
        source_name: str,
        target_name: str,
        **kwargs: object,
    ) -> None:
        nonlocal foreign_anchor
        calls["atomic"] += 1
        output_status = os.fstat(output_fd)
        output_anchor = (output_status.st_dev, output_status.st_ino)
        production_atomic(
            output_fd, source_name, target_name, **kwargs
        )
        original_name = f".original-{target_name}"
        real_rename(
            target_name,
            original_name,
            src_dir_fd=output_fd,
            dst_dir_fd=output_fd,
        )
        os.mkdir(target_name, dir_fd=output_fd)
        foreign_fd = os.open(
            target_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=output_fd,
        )
        try:
            marker_fd = os.open(
                "foreign-marker",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=foreign_fd,
            )
            try:
                os.write(marker_fd, marker_bytes)
            finally:
                os.close(marker_fd)
        finally:
            os.close(foreign_fd)
        status = os.stat(
            target_name, dir_fd=output_fd, follow_symlinks=False
        )
        foreign_anchor = (status.st_dev, status.st_ino)

        def uses_output_directory(kwargs: dict[str, object]) -> bool:
            descriptor = kwargs.get(
                "src_dir_fd", kwargs.get("dir_fd")
            )
            if not isinstance(descriptor, int):
                return False
            try:
                status = os.fstat(descriptor)
            except OSError:
                return False
            return (status.st_dev, status.st_ino) == output_anchor

        def record_rename(*args: object, **kwargs: object) -> None:
            if uses_output_directory(kwargs):
                calls["rename"] += 1
                return
            real_rename(*args, **kwargs)

        def record_unlink(*args: object, **kwargs: object) -> None:
            if uses_output_directory(kwargs):
                calls["unlink"] += 1
                return
            real_unlink(*args, **kwargs)

        def record_rmdir(*args: object, **kwargs: object) -> None:
            if uses_output_directory(kwargs):
                calls["rmdir"] += 1
                return
            real_rmdir(*args, **kwargs)

        monkeypatch.setattr(dashboard.os, "rename", record_rename)
        monkeypatch.setattr(dashboard.os, "unlink", record_unlink)
        monkeypatch.setattr(dashboard.os, "rmdir", record_rmdir)

    monkeypatch.setattr(
        dashboard, "_atomic_rename_noreplace", publish_then_replace
    )

    with pytest.raises(
        dashboard.X13DashboardExportError,
        match=(
            "published but verification failed.*manual inspection required"
        ),
    ):
        dashboard.export_x13_dashboard(batch, output)

    target = output / f"batch={BATCH_ID.replace(':', '-')}"
    status = target.stat()
    assert foreign_anchor is not None
    assert (status.st_dev, status.st_ino) == foreign_anchor
    assert (target / "foreign-marker").read_bytes() == marker_bytes
    assert calls == {
        "atomic": 1,
        "rename": 0,
        "unlink": 0,
        "rmdir": 0,
    }


def test_foreign_stage_substitute_is_left_untouched_as_blocking_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    output = tmp_path / "output"
    original = dashboard._atomic_rename_noreplace
    substitute: dict[str, object] = {}
    replaced = False

    def replace_stage_then_rename_substitute(
        output_fd: int,
        source_name: str,
        target_name: str,
        **kwargs: object,
    ) -> None:
        nonlocal replaced
        if replaced:
            original(output_fd, source_name, target_name, **kwargs)
            return
        replaced = True
        backup = f".original-{source_name.removeprefix('.')}"
        os.rename(
            source_name,
            backup,
            src_dir_fd=output_fd,
            dst_dir_fd=output_fd,
        )
        os.mkdir(source_name, dir_fd=output_fd)
        substitute_fd = os.open(
            source_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=output_fd,
        )
        try:
            marker_fd = os.open(
                "substitute-marker",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=substitute_fd,
            )
            try:
                os.write(marker_fd, b"preserve")
            finally:
                os.close(marker_fd)
        finally:
            os.close(substitute_fd)
        status = os.stat(
            source_name, dir_fd=output_fd, follow_symlinks=False
        )
        substitute.update(
            name=source_name, anchor=(status.st_dev, status.st_ino)
        )
        original(output_fd, source_name, target_name)

    monkeypatch.setattr(
        dashboard,
        "_atomic_rename_noreplace",
        replace_stage_then_rename_substitute,
    )
    with pytest.raises(
        dashboard.X13DashboardExportError,
        match=(
            "published but verification failed.*manual inspection required"
        ),
    ):
        dashboard.export_x13_dashboard(batch, output)

    target = output / f"batch={BATCH_ID.replace(':', '-')}"
    status = target.stat()
    assert (status.st_dev, status.st_ino) == substitute["anchor"]
    assert (target / "substitute-marker").read_bytes() == b"preserve"
    monkeypatch.setattr(dashboard, "_atomic_rename_noreplace", original)
    with pytest.raises(
        dashboard.X13DashboardExportError, match="target already exists"
    ):
        dashboard.export_x13_dashboard(batch, output)


def test_unanchored_foreign_symlink_target_is_left_completely_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    output = tmp_path / "output"
    outside = tmp_path / "outside-substitute"
    outside.mkdir()
    original = dashboard._atomic_rename_noreplace
    replaced = False
    substitute_name = ""

    def replace_published_target_with_symlink(
        output_fd: int,
        source_name: str,
        target_name: str,
        **kwargs: object,
    ) -> None:
        nonlocal replaced, substitute_name
        original(output_fd, source_name, target_name, **kwargs)
        if replaced:
            return
        replaced = True
        substitute_name = source_name
        os.rename(
            target_name,
            f".original-{target_name}",
            src_dir_fd=output_fd,
            dst_dir_fd=output_fd,
        )
        os.symlink(outside, target_name, dir_fd=output_fd)

    monkeypatch.setattr(
        dashboard,
        "_atomic_rename_noreplace",
        replace_published_target_with_symlink,
    )
    with pytest.raises(dashboard.X13DashboardExportError):
        dashboard.export_x13_dashboard(batch, output)

    target = output / f"batch={BATCH_ID.replace(':', '-')}"
    assert substitute_name
    before = target.lstat()
    assert target.is_symlink()
    assert target.readlink() == outside
    after = target.lstat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    assert list(outside.iterdir()) == []


def test_asset_over_limit_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)
    monkeypatch.setattr(dashboard, "MAX_ASSET_BYTES", 128)

    with pytest.raises(dashboard.X13DashboardExportError, match="8 MiB|limit"):
        dashboard.export_x13_dashboard(batch, tmp_path / "output")
    assert not (tmp_path / "output" / f"batch={BATCH_ID.replace(':', '-')}").exists()


def test_cli_exports_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = _write_batch(tmp_path)
    _stub_batch_verifier(monkeypatch)

    assert (
        dashboard.main(
            ["--batch", str(batch), "--output", str(tmp_path / "output")]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["batch_id"] == BATCH_ID
    assert output["publish_manifest_sha256"].startswith("sha256:")
