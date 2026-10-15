#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

if ! command -v tide-download-data >/dev/null 2>&1; then
  echo "tide-download-data is not available in the active Python environment." >&2
  exit 1
fi

tide-download-data --source healthbench --limit "${HEALTHBENCH_SAMPLE_LIMIT:-100}"
tide-download-data --source wildchat --limit "${WILDCHAT_SAMPLE_LIMIT:-100}"
