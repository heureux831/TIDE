#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${TIDE_USE_CURRENT_ENV:-0}" == "1" ]]; then
  PYTHON_BIN="${TIDE_PYTHON_BIN:-python}"
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -c \
    'import torch; assert torch.cuda.is_available(), "current PyTorch cannot access CUDA"'
  "${PYTHON_BIN}" -m pip install -e '.[dev]'
else
  python3.12 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip

  # PyTorch's official CUDA 12.8 wheel is the reproducible clean-environment
  # baseline. Do not silently change this index.
  python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  python -m pip install -e '.[dev]'
fi

"${PYTHON_BIN:-python}" -m pip check

echo "Environment ready. Fill .env, then run: tide preflight --deep"
