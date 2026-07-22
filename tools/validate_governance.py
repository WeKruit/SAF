"""Validate the research program's governed source files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from prediction_market.governance import validate_program


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    arguments = parser.parse_args(argv)
    report = validate_program(arguments.root)

    if report.is_valid:
        print(
            f"OK: {report.source_count} sources, "
            f"{report.catalog_count} catalog items, "
            f"{report.assignment_count} assignments"
        )
        return 0

    print(f"INVALID: {len(report.violations)} governance violation(s)")
    for violation in report.violations:
        print(f"{violation.code}: {violation.path}: {violation.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
