#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${1:-0}"
if [[ "${GPU_ID}" == "-h" || "${GPU_ID}" == "--help" ]]; then
  exec "${PYTHON_BIN:-python}" "${PROJECT_ROOT}/scripts/f1b/run_group.py" --group mopf-nc --help
fi
if [[ $# -gt 0 ]]; then
  shift
fi

exec "${PYTHON_BIN:-python}" "${PROJECT_ROOT}/scripts/f1b/run_group.py" \
  --group mopf-nc \
  --gpu-id "${GPU_ID}" \
  "$@"
