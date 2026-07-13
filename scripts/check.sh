#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors
#
# Local check. Mirrors the GitHub Actions workflow (.github/workflows/ci.yml)
# `check` job: ruff lint, ruff format, mypy, pytest. Keep the two in sync.
#
# Usage (from anywhere):  bash scripts/check.sh
# Exit code is non-zero if any check fails (so it can block a merge).
set -euo pipefail

# Run from the jaxterity project root regardless of caller's CWD.
cd "$(dirname "$0")/.."

run() {
  echo ""
  echo "== $1 =="
  shift
  "$@"
}

run "ruff check"          python -m ruff check .
run "ruff format --check" python -m ruff format --check .
run "mypy jaxterity/"     python -m mypy jaxterity/
# Mirror ci.yml's PR `check` job: exclude the nightly long-form sweep (run it
# with `pytest -m nightly`, as the ci.yml `nightly` job does). `hardware` tests
# need a physical servo bus and `pi` tests assert real-wall-clock timing on a
# specific host; both are excluded here (run them on the bench / target).
run "pytest"              python -m pytest test -m "not nightly and not hardware and not pi and not render" -q

echo ""
echo "All checks passed (PR tier; run 'pytest -m nightly' for the long-form sweep)."
