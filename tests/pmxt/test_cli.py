from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "pmxt"


def test_inventory_cli_writes_evidence_from_fetched_entries(tmp_path, monkeypatch):
    from prediction_market.cli import audit_pmxt
    from prediction_market.pmxt.archive import parse_inventory_page

    entries = parse_inventory_page(
        (FIXTURE_ROOT / "archive_index.html").read_text(encoding="utf-8"),
        index_url="https://archive.pmxt.dev/Polymarket/v2",
    )
    monkeypatch.setattr(
        audit_pmxt,
        "fetch_archive_inventory",
        lambda **_kwargs: entries,
    )
    output = tmp_path / "inventory.json"

    exit_code = audit_pmxt.main(["inventory", "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["audit_kind"] == "pmxt_phase0_inventory"
    assert report["summary"]["covered_hours"] == 2
    assert len(report["objects"]) == 2
    assert report["source_url"] == "https://archive.pmxt.dev/Polymarket/v2"


def test_sample_cli_honors_max_files_and_reports_original_hash(tmp_path, monkeypatch):
    from prediction_market.cli import audit_pmxt
    from prediction_market.pmxt.archive import ArchivedFile, parse_inventory_page

    entries = parse_inventory_page(
        (FIXTURE_ROOT / "archive_index.html").read_text(encoding="utf-8"),
        index_url="https://archive.pmxt.dev/Polymarket/v2",
    )
    monkeypatch.setattr(
        audit_pmxt, "fetch_archive_inventory", lambda **_kwargs: entries
    )
    calls: list[str] = []

    def fake_download(url, *, raw_root, max_bytes, timeout_seconds):
        calls.append(url)
        object_path = Path(raw_root) / "object.parquet"
        object_path.parent.mkdir(parents=True, exist_ok=True)
        object_path.write_bytes(b"real response bytes supplied by test transport")
        return ArchivedFile(
            source_filename=Path(url).name,
            source_url=url,
            sha256="sha256:" + "a" * 64,
            byte_size=45,
            object_path=object_path,
        )

    monkeypatch.setattr(audit_pmxt, "download_and_preserve", fake_download)
    monkeypatch.setattr(
        audit_pmxt,
        "audit_parquet",
        lambda path: SimpleNamespace(
            to_dict=lambda: {
                "path": str(path),
                "original_sha256": "sha256:" + "a" * 64,
                "row_count": 1,
            }
        ),
    )
    output = tmp_path / "sample.json"

    exit_code = audit_pmxt.main(
        [
            "sample",
            "--max-files",
            "1",
            "--raw-root",
            str(tmp_path / "raw"),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(calls) == 1
    assert len(report["files"]) == 1
    assert report["files"][0]["archive"]["sha256"] == "sha256:" + "a" * 64
    assert report["queue_fill_reconstructed"] is False


def test_sample_cli_returns_explicit_failure_without_fabricating_output(
    tmp_path, monkeypatch, capsys
):
    from prediction_market.cli import audit_pmxt
    from prediction_market.pmxt.archive import ArchiveNetworkError, parse_inventory_page

    entries = parse_inventory_page(
        (FIXTURE_ROOT / "archive_index.html").read_text(encoding="utf-8"),
        index_url="https://archive.pmxt.dev/Polymarket/v2",
    )
    monkeypatch.setattr(
        audit_pmxt, "fetch_archive_inventory", lambda **_kwargs: entries
    )

    def fail_download(*_args, **_kwargs):
        raise ArchiveNetworkError("network sample unavailable")

    monkeypatch.setattr(audit_pmxt, "download_and_preserve", fail_download)
    output = tmp_path / "must-not-exist.json"

    exit_code = audit_pmxt.main(
        [
            "sample",
            "--max-files",
            "1",
            "--raw-root",
            str(tmp_path / "raw"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert "network sample unavailable" in capsys.readouterr().err
    assert not output.exists()
