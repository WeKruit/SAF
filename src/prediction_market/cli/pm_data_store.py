"""Mirror PMXT immutable data objects to a private Supabase S3 bucket."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from prediction_market.pmxt.data_store import (
    DataStoreError,
    configured_object_store,
    load_remote_inventory,
    prune_pmxt_local,
    status_pmxt,
    upload_pmxt,
    verify_pmxt_remote,
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


def _add_common_s3_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-file",
        type=Path,
        help="S3 env file; defaults to .env.s3 or .env in the current directory",
    )
    parser.add_argument("--output", type=Path)


def _store_and_prefix(args: argparse.Namespace):
    return configured_object_store(env_file=args.env_file)


def _upload_report(args: argparse.Namespace) -> dict[str, object]:
    store, prefix = _store_and_prefix(args)
    return upload_pmxt(args.raw_root, store, prefix=prefix)


def _verify_report(args: argparse.Namespace) -> dict[str, object]:
    store, prefix = _store_and_prefix(args)
    return verify_pmxt_remote(store, prefix=prefix)


def _status_report(args: argparse.Namespace) -> dict[str, object]:
    remote_inventory = load_remote_inventory(args.remote_inventory)
    manifest_roots = tuple(
        args.manifest_root or [Path("artifacts"), Path("var/raw/manifests")]
    )
    return status_pmxt(
        args.raw_root,
        prefix=args.prefix,
        manifest_roots=manifest_roots,
        remote_inventory=remote_inventory,
    )


def _prune_report(args: argparse.Namespace) -> dict[str, object]:
    return prune_pmxt_local(
        args.raw_root,
        remote_inventory=load_remote_inventory(args.remote_inventory),
        execute=args.execute,
        confirm_no_local_readers=args.confirm_no_local_readers,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pm-data-store",
        description="Mirror and verify immutable PMXT Parquet objects in Supabase S3.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload = subparsers.add_parser(
        "upload-pmxt",
        help="upload local canonical PMXT objects to Supabase S3",
    )
    _add_common_s3_args(upload)
    upload.add_argument("--raw-root", type=Path, default=Path("var/raw"))

    verify = subparsers.add_parser(
        "verify-pmxt",
        help="stream remote PMXT objects and recompute SHA-256",
    )
    _add_common_s3_args(verify)

    status = subparsers.add_parser(
        "status-pmxt",
        help="compare local canonical objects, remote inventory, and manifest refs",
    )
    status.add_argument("--output", type=Path)
    status.add_argument("--raw-root", type=Path, default=Path("var/raw"))
    status.add_argument(
        "--manifest-root",
        action="append",
        type=Path,
    )
    status.add_argument(
        "--remote-inventory",
        type=Path,
        required=True,
        help="verified inventory JSON from verify-pmxt",
    )
    status.add_argument(
        "--prefix",
        default="prediction-market",
        help="S3 prefix used only with --remote-inventory",
    )

    prune = subparsers.add_parser(
        "prune-pmxt-local",
        help="delete local PMXT files only after a verified remote inventory exists",
    )
    prune.add_argument("--raw-root", type=Path, default=Path("var/raw"))
    prune.add_argument("--remote-inventory", type=Path, required=True)
    prune.add_argument("--output", type=Path)
    prune.add_argument("--execute", action="store_true")
    prune.add_argument(
        "--confirm-no-local-readers",
        help="required with --execute; value must be NO_LOCAL_PMXT_READERS",
    )
    return parser


def _report_exit_code(report: dict[str, object]) -> int:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        return 0
    if report.get("audit_kind") in {"pmxt_s3_upload_v1", "pmxt_s3_verify_v1"}:
        return 2 if int(summary.get("failure_count", 0)) else 0
    if report.get("audit_kind") == "pmxt_s3_status_v1":
        for key in (
            "local_missing_remote_count",
            "manifest_missing_local_count",
            "manifest_missing_remote_count",
            "remote_missing_local_count",
        ):
            if int(summary.get(key, 0)):
                return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "upload-pmxt":
            report = _upload_report(args)
        elif args.command == "verify-pmxt":
            report = _verify_report(args)
        elif args.command == "status-pmxt":
            report = _status_report(args)
        else:
            report = _prune_report(args)
        _write_report(report, args.output)
        return _report_exit_code(report)
    except (DataStoreError, OSError, ValueError) as exc:
        print(f"pm-data-store failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
