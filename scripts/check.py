"""Project gate: lint + format check + type check + tests with branch coverage.

Run before pushing or opening a PR::

    uv run python scripts/check.py

The dev loop is just ``uv run pytest`` (fast, no coverage gate). This script is
the enforcement layer: it fails if ruff is unhappy, formatting would change,
mypy finds a type error, or branch coverage on ``imageProcToolkit`` drops below
the threshold below.
"""

from __future__ import annotations

import subprocess
import sys

# Coverage target is the package import name. Find-replace when renaming the
# package (same target as pyproject.toml `packages = [...]` and the tests).
PACKAGE = "imageProcToolkit"
COVERAGE_FAIL_UNDER = 90


def _run(label: str, cmd: list[str]) -> bool:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)  # noqa: S603 -- intentional, controlled args
    ok = result.returncode == 0
    print(f"--- {label}: {'PASS' if ok else 'FAIL'} ---")
    return ok


def main() -> int:
    py = sys.executable
    steps = [
        ("ruff check", [py, "-m", "ruff", "check", "."]),
        ("ruff format --check", [py, "-m", "ruff", "format", "--check", "."]),
        ("mypy", [py, "-m", "mypy", f"src/{PACKAGE}"]),
        (
            "pytest + coverage",
            [
                py,
                "-m",
                "pytest",
                "--cov",
                PACKAGE,
                "--cov-branch",
                "--cov-report",
                "term-missing",
                "--cov-fail-under",
                str(COVERAGE_FAIL_UNDER),
            ],
        ),
    ]
    for label, cmd in steps:
        if not _run(label, cmd):
            print("\n=== GATE: FAIL (stopped at " + label + ") ===")
            return 1
    print("\n=== GATE: PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
