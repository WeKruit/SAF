"""Audit the public PMXT v2 archive without inventing unavailable evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from prediction_market.pmxt.archive import (
    PMXT_V2_INDEX_URL,
    ArchiveEntry,
    ArchiveError,
    audit_parquet,
    download_and_preserve,
    fetch_archive_inventory,
    summarize_inventory,
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _write_report(report: dict[str, object], output: Path | None) -> None:
    rendered = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    if output is None:
        sys.stdout.write(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"wrote {output}")


def _inventory_report(args: argparse.Namespace) -> dict[str, object]:
    entries = fetch_archive_inventory(
        index_url=args.index_url,
        timeout_seconds=args.timeout_seconds,
        max_pages=args.max_pages,
    )
    return {
        "audit_kind": "pmxt_phase0_inventory",
        "retrieved_at": _utc_now(),
        "source_url": args.index_url,
        "partial_page_limit": args.max_pages,
        "summary": summarize_inventory(entries),
        "objects": [entry.to_dict() for entry in entries],
    }


def _sample_candidates(args: argparse.Namespace) -> tuple[ArchiveEntry, ...]:
    if args.url:
        entries = [
            ArchiveEntry(
                filename=Path(url).name,
                url=url,
                hour=datetime.strptime(
                    Path(url)
                    .name.removeprefix("polymarket_orderbook_")
                    .removesuffix(".parquet"),
                    "%Y-%m-%dT%H",
                ).replace(tzinfo=timezone.utc),
                size_bytes=None,
            )
            for url in args.url
        ]
    else:
        entries = list(
            fetch_archive_inventory(
                index_url=args.index_url,
                timeout_seconds=args.timeout_seconds,
            )
        )
    entries.sort(
        key=lambda entry: (
            entry.size_bytes is None,
            entry.size_bytes if entry.size_bytes is not None else 0,
            -entry.hour.timestamp(),
            entry.filename,
        )
    )
    return tuple(entries[: args.max_files])


def _sample_report(args: argparse.Namespace) -> dict[str, object]:
    candidates = _sample_candidates(args)
    if not candidates:
        raise ArchiveError("no PMXT files were selected for sampling")
    files: list[dict[str, object]] = []
    for entry in candidates:
        archived = download_and_preserve(
            entry.url,
            raw_root=args.raw_root,
            max_bytes=args.max_bytes,
            timeout_seconds=args.timeout_seconds,
        )
        parquet_audit = audit_parquet(archived.object_path)
        files.append(
            {
                "inventory_object": entry.to_dict(),
                "archive": archived.to_dict(),
                "parquet_audit": parquet_audit.to_dict(),
            }
        )
    return {
        "audit_kind": "pmxt_phase1_public_sample",
        "retrieved_at": _utc_now(),
        "source_kind": "real_public_https_download",
        "max_files": args.max_files,
        "files": files,
        "queue_fill_reconstructed": False,
        "limitations": [
            "PMXT v2 has no exchange sequence number.",
            "This command audits original Parquet bytes and schema; it does not infer queue position or fills.",
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit-pmxt",
        description="Inventory and sample the public PMXT v2 archive.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser(
        "inventory", help="inventory public hourly objects"
    )
    inventory.add_argument("--index-url", default=PMXT_V2_INDEX_URL)
    inventory.add_argument("--max-pages", type=_positive_integer)
    inventory.add_argument("--timeout-seconds", type=_positive_float, default=30.0)
    inventory.add_argument("--output", type=Path)

    sample = subparsers.add_parser(
        "sample", help="download, preserve, and audit a bounded real sample"
    )
    sample.add_argument("--index-url", default=PMXT_V2_INDEX_URL)
    sample.add_argument(
        "--url", action="append", help="explicit PMXT hourly object URL"
    )
    sample.add_argument("--max-files", type=_positive_integer, default=1)
    sample.add_argument("--max-bytes", type=_positive_integer, default=600_000_000)
    sample.add_argument("--timeout-seconds", type=_positive_float, default=120.0)
    sample.add_argument("--raw-root", type=Path, required=True)
    sample.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            report = _inventory_report(args)
        else:
            report = _sample_report(args)
        _write_report(report, args.output)
    except (ArchiveError, OSError, ValueError) as exc:
        print(f"audit-pmxt failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
