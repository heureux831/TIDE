#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

if ! command -v tide-download-models >/dev/null 2>&1; then
  echo "tide-download-models is not available in the active Python environment." >&2
  exit 1
fi

tide-download-models --model flan-t5 --provider "${TIDE_MODEL_PROVIDER:-auto}"
tide-download-models --model qwen --provider "${TIDE_MODEL_PROVIDER:-auto}"
