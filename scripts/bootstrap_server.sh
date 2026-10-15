#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# PyTorch's official CUDA 12.8 wheel is the minimum documented Blackwell path
# selected by the experiment specification. Do not silently change this index.
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
python -m pip check

echo "Environment ready. Fill .env, then run: tide preflight --deep"
