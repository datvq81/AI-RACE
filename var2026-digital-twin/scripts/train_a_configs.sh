#!/usr/bin/env bash

# Backward-compatible wrapper for the generic JSON experiment-suite runner.
# Keep this filename for old commands; edit/copy the suite JSON, not this file.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SUITE="${SUITE:-${PROJECT_ROOT}/configs/experiments/a_baseline.json}"
STAGE="${STAGE:-train}"

command=(
    "$PYTHON_BIN" "${SCRIPT_DIR}/run_experiment_suite.py"
    --suite "$SUITE"
    --stage "$STAGE"
)

# Optional environment-variable overrides preserve the old invocation style.
[[ -n "${SCENE:-}" ]] && command+=(--scene "$SCENE")
[[ -n "${ITERATIONS:-}" ]] && command+=(--iterations "$ITERATIONS")
[[ -n "${SEED:-}" ]] && command+=(--seed "$SEED")

cd "$PROJECT_ROOT"
exec "${command[@]}" "$@"
