"""Export and privately publish X-13 dashboard presentation assets."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from prediction_market.dashboard_store import (
    DashboardStoreError,
    load_dashboard_s3_config,
    publish_dashboard_export,
    verify_dashboard_publication,
)
from prediction_market.sports.nfl_x13_dashboard import (
    X13DashboardExportError,
    export_x13_dashboard,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="publish-x13-dashboard",
        description="Export, publish, or verify private X-13 dashboard assets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser(
        "export", help="build a local verified dashboard projection"
    )
    export.add_argument("--batch", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    publish = commands.add_parser(
        "publish-s3", help="publish an existing verified projection to S3"
    )
    publish.add_argument("--export-root", type=Path, required=True)
    publish.add_argument("--env-file", type=Path)
    publish.add_argument(
        "--workers",
        type=int,
        default=8,
        help="bounded immutable-content upload workers (1-32; default: 8)",
    )

    verify = commands.add_parser(
        "verify-s3", help="verify the current private S3 publication"
    )
    verify.add_argument("--env-file", type=Path)
    return parser


def _render(value: object) -> None:
    print(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "export":
            result = export_x13_dashboard(
                arguments.batch, arguments.output
            )
            _render(
                {
                    "batch_id": result.batch_id,
                    "publish_root": str(result.publish_root),
                    "manifest_path": str(result.manifest_path),
                    "publish_manifest_sha256": (
                        result.publish_manifest_sha256
                    ),
                    "game_count": result.game_count,
                    "asset_count": result.asset_count,
                }
            )
        else:
            config = load_dashboard_s3_config(
                env_file=arguments.env_file
            )
            if arguments.command == "publish-s3":
                result = publish_dashboard_export(
                    arguments.export_root,
                    config,
                    max_workers=arguments.workers,
                )
            else:
                result = verify_dashboard_publication(config)
            _render(asdict(result))
        return 0
    except (
        DashboardStoreError,
        X13DashboardExportError,
        OSError,
        ValueError,
    ) as error:
        print(
            f"publish-x13-dashboard failed: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
