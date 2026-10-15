#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

tide-download-data --source healthbench --limit "${HEALTHBENCH_SAMPLE_LIMIT:-100}"
tide-download-data --source wildchat --limit "${WILDCHAT_SAMPLE_LIMIT:-100}"
